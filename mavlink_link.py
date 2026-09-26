"""MAVLink link to ArduPilot (SpeedyBee F405) over the Pi's UART.

Threads
-------
* rx thread   - blocking ``recv_match`` loop, decodes telemetry into a
                ``VehicleState`` snapshot and records COMMAND_ACKs.
* hb thread   - sends a companion-computer HEARTBEAT at 1 Hz so the FC (and
                any GCS) can see the onboard computer is alive.

Control is done in GUIDED mode with SET_POSITION_TARGET_LOCAL_NED velocity
setpoints streamed at 10 Hz. ArduPilot stops the vehicle if no setpoint
arrives for ~3 s (GUID_TIMEOUT), which is the first line of defence if the
Pi software hangs.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from atlas.comms.base import FlightLink
from atlas.core.geo import LatLon
from atlas.core.state import VehicleState

log = logging.getLogger("atlas.mavlink")

try:  # pymavlink is only required on the vehicle / with SITL
    from pymavlink import mavutil
except ImportError:  # pragma: no cover
    mavutil = None

# SET_POSITION_TARGET type_mask bits
_IGNORE_POS = 0b0000_0000_0111
_IGNORE_VEL = 0b0000_0011_1000
_IGNORE_ACC = 0b0001_1100_0000
_IGNORE_YAW = 0b0100_0000_0000
_IGNORE_YAW_RATE = 0b1000_0000_0000
MASK_VEL_YAW = _IGNORE_POS | _IGNORE_ACC | _IGNORE_YAW_RATE          # velocity + absolute yaw
MASK_VEL_HOLD_YAW = _IGNORE_POS | _IGNORE_ACC | _IGNORE_YAW          # velocity + yaw_rate (=0)


class MavlinkLink(FlightLink):
    def __init__(self, cfg, clock=None):
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed: pip install pymavlink")
        self.cfg = cfg
        self._clock_now = clock.now if clock else time.monotonic
        self._master = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._ack_cv = threading.Condition()
        self._acks: dict[int, int] = {}
        self._home: Optional[LatLon] = None
        self._boot = time.monotonic()
        # raw telemetry fields, merged into a VehicleState on each update
        self._t = dict(lat=0.0, lon=0.0, alt_rel=0.0, vn=0.0, ve=0.0, vd=0.0,
                       roll=0.0, pitch=0.0, yaw=0.0, armed=False, mode="UNKNOWN",
                       battery_v=0.0, battery_pct=-1.0, gps_fix=0, satellites=0,
                       hdop=99.0, fc_heartbeat=0.0)
        self._state: Optional[VehicleState] = None
        self._rc: dict[int, int] = {}
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ setup
    def connect(self, timeout: float = 30.0) -> None:
        c = self.cfg.mavlink
        log.info("connecting to %s @ %d", c.connection, c.baud)
        self._master = mavutil.mavlink_connection(
            c.connection, baud=c.baud,
            source_system=c.source_system, source_component=c.source_component,
            autoreconnect=True,
        )
        hb = self._master.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise TimeoutError("no heartbeat from flight controller")
        log.info("FC heartbeat: sys=%d comp=%d", self._master.target_system, self._master.target_component)
        self._request_streams(c.stream_hz)
        self._request_home()
        for name, fn in (("mav-rx", self._rx_loop), ("mav-hb", self._hb_loop)):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(1.0)
        if self._master:
            self._master.close()

    def _request_streams(self, hz: float) -> None:
        m = self._master.mav
        interval_us = int(1e6 / hz)
        for msg_id in (
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
        ):
            self._command_long(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, interval_us, wait=False)
        for msg_id in (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS):
            self._command_long(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 200_000, wait=False)
        _ = m  # keep reference for readability

    def _request_home(self) -> None:
        self._command_long(mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
                           mavutil.mavlink.MAVLINK_MSG_ID_HOME_POSITION, wait=False)

    # --------------------------------------------------------------- threads
    def _hb_loop(self) -> None:
        period = 1.0 / self.cfg.mavlink.heartbeat_hz
        while not self._stop.wait(period):
            try:
                self._master.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0,
                    mavutil.mavlink.MAV_STATE_ACTIVE)
            except Exception as exc:  # serial hiccup; autoreconnect handles it
                log.warning("heartbeat send failed: %s", exc)

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._master.recv_match(blocking=True, timeout=0.5)
            except Exception as exc:
                log.warning("recv error: %s", exc)
                continue
            if msg is None:
                continue
            self._handle(msg)

    def _handle(self, msg) -> None:
        mtype = msg.get_type()
        t = self._t
        now = self._clock_now()
        if mtype == "HEARTBEAT":
            if (msg.get_srcSystem() != self._master.target_system
                    or msg.type == mavutil.mavlink.MAV_TYPE_GCS
                    or msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID):
                return  # ignore GCS, gimbals, and our own companion heartbeat
            t["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            t["mode"] = mavutil.mode_string_v10(msg)
            t["fc_heartbeat"] = now
        elif mtype == "GLOBAL_POSITION_INT":
            t["lat"], t["lon"] = msg.lat / 1e7, msg.lon / 1e7
            t["alt_rel"] = msg.relative_alt / 1000.0
            t["vn"], t["ve"], t["vd"] = msg.vx / 100.0, msg.vy / 100.0, msg.vz / 100.0
            t["yaw"] = (msg.hdg / 100.0) if msg.hdg != 65535 else t["yaw"]
        elif mtype == "ATTITUDE":
            import math
            t["roll"], t["pitch"] = math.degrees(msg.roll), math.degrees(msg.pitch)
            t["yaw"] = (math.degrees(msg.yaw) + 360.0) % 360.0
        elif mtype == "SYS_STATUS":
            t["battery_v"] = msg.voltage_battery / 1000.0
            t["battery_pct"] = float(msg.battery_remaining)
        elif mtype == "GPS_RAW_INT":
            t["gps_fix"], t["satellites"] = msg.fix_type, msg.satellites_visible
            t["hdop"] = msg.eph / 100.0 if msg.eph != 65535 else 99.0
        elif mtype == "RC_CHANNELS":
            self._rc = {i: getattr(msg, f"chan{i}_raw") for i in range(1, 19)}
            return
        elif mtype == "HOME_POSITION":
            self._home = LatLon(msg.latitude / 1e7, msg.longitude / 1e7)
            log.info("home set to %.7f, %.7f", self._home.lat, self._home.lon)
        elif mtype == "COMMAND_ACK":
            with self._ack_cv:
                self._acks[msg.command] = msg.result
                self._ack_cv.notify_all()
            return
        elif mtype == "STATUSTEXT":
            log.info("FC: %s", msg.text)
            return
        else:
            return
        with self._lock:
            self._state = VehicleState(stamp=now, **t)

    # --------------------------------------------------------------- queries
    def state(self) -> Optional[VehicleState]:
        with self._lock:
            return self._state

    def authorized(self) -> bool:
        c = self.cfg.mavlink
        return self._rc.get(c.auth_rc_channel, 0) >= c.auth_pwm

    def home(self) -> Optional[LatLon]:
        if self._home is None:
            self._request_home()
        return self._home

    # -------------------------------------------------------------- commands
    def _command_long(self, command, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0,
                      wait: bool = True, timeout: float = 3.0) -> bool:
        with self._ack_cv:
            self._acks.pop(command, None)
        self._master.mav.command_long_send(
            self._master.target_system, self._master.target_component,
            command, 0, p1, p2, p3, p4, p5, p6, p7)
        if not wait:
            return True
        deadline = time.monotonic() + timeout
        with self._ack_cv:
            while command not in self._acks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("no ACK for command %d", command)
                    return False
                self._ack_cv.wait(remaining)
            ok = self._acks[command] == mavutil.mavlink.MAV_RESULT_ACCEPTED
        if not ok:
            log.warning("command %d rejected (result %d)", command, self._acks[command])
        return ok

    def set_mode(self, mode: str) -> bool:
        mapping = self._master.mode_mapping() or {}
        if mode not in mapping:
            log.error("unknown mode %s", mode)
            return False
        return self._command_long(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                                  mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mapping[mode])

    def arm(self) -> bool:
        return self._command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, timeout=5.0)

    def disarm(self) -> bool:
        return self._command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0)

    def takeoff(self, alt: float) -> bool:
        return self._command_long(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt)

    def send_velocity(self, vn: float, ve: float, vd: float, yaw: Optional[float] = None) -> None:
        import math
        boot_ms = int((time.monotonic() - self._boot) * 1000) & 0xFFFFFFFF
        if yaw is None:
            mask, yaw_rad, yaw_rate = MASK_VEL_HOLD_YAW, 0.0, 0.0
        else:
            mask, yaw_rad, yaw_rate = MASK_VEL_YAW, math.radians(yaw), 0.0
        self._master.mav.set_position_target_local_ned_send(
            boot_ms, self._master.target_system, self._master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, mask,
            0, 0, 0, vn, ve, vd, 0, 0, 0, yaw_rad, yaw_rate)

    def set_servo(self, channel: int, pwm: int) -> bool:
        return self._command_long(mavutil.mavlink.MAV_CMD_DO_SET_SERVO, channel, pwm)
