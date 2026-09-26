# ATLAS: autonomous delivery drone onboard software

Onboard software for ATLAS, a quadcopter that flies a GPS route, avoids obstacles
reactively, finds a drop-zone marker with an on-device CNN and releases a payload
on it. Everything runs on a Raspberry Pi 4 that commands an ArduPilot flight
controller (SpeedyBee F405) over MAVLink.

| Subsystem | What it does | Key numbers |
|---|---|---|
| Navigation | GUIDED-mode NED velocity setpoints over MAVLink, yaw-first waypoint guidance | 10 Hz setpoints, 2.5 m/s cruise |
| Obstacle avoidance | 3× HC-SR04 + 4× VL53L1X fused into 5 horizontal sectors, speed governor, replanning ladder | 10 Hz, trigger 2.5 m, critical 1.0 m |
| Perception | INT8 MobileNetV2 (α 0.5, 160 px) localiser → N-of-M confirmation → ground projection | 15 fps target on Pi 4 |
| Delivery | expanding-square search, PI visual servo, down-ToF descent, servo release | release at 3 m AGL |
| Safety | preflight gate, in-flight failsafe arbiter, watchdog, pilot MISSION-GO switch, FC failsafes underneath | |

## Repository

```
atlas/
  app.py                 entry point: run | sitl | check
  core/                  config, geodesy, clocks & rate loops, shared data types
  comms/                 FlightLink interface, MAVLink implementation, kinematic sim
  sensors/               HC-SR04 (pigpio), VL53L1X (XSHUT addressing), fusion array, sim sensors
  avoidance/reactive.py  speed governor, corridor test, replan ladder, side repulsion
  navigation/            mission file, route, expanding-square search, waypoint guidance
  perception/            camera, TFLite detector, projection geometry, tracker, pipeline, sim detector
  payload/release.py     servo release through the FC
  safety/                monitor (preflight + failsafes), geofence, watchdog + systemd notify
  mission/executive.py   10 Hz state machine that ties it all together
  telemetry/blackbox.py  CSV + JSONL flight recorder
sim/                     closed-loop SIL harness, worlds, missions, Monte-Carlo sweep
training/                synthetic data, YOLO→CSV, train, INT8 quantise + verify, Pi benchmark
tests/                   70 unit, integration and closed-loop mission tests
deploy/                  Pi setup script, systemd unit, ArduPilot parameter file
tools/                   sensor bench view, blackbox plotter
config/atlas.yaml        every tunable in one place
```

## Quick start (any PC, no hardware)

```bash
pip install -r requirements.txt
python -m pytest                              # 70 tests, ~10 s
python -m sim.run_sim --plot demo.png         # full simulated delivery
python -m sim.run_sim --fault low_battery     # also: gps_loss, front_sensor_fail, no_marker, wind
python -m sim.monte_carlo --runs 50           # randomised robustness sweep
```

## ArduPilot SITL (real autopilot, simulated sensors)

```bash
# terminal 1: ArduPilot SITL at the mission's home
sim_vehicle.py -v ArduCopter --custom-location=12.9716,77.5946,900,0 --no-mavproxy
# terminal 2: ATLAS against SITL, with ray-cast sensors and synthetic detector
python -m atlas sitl --mission sim/missions/demo.yaml --connection tcp:127.0.0.1:5760
```
Set RC7 high in SITL (`rc 7 2000` in MAVProxy) to give MISSION-GO.

## On the vehicle

```bash
bash deploy/setup_pi.sh                        # once; then reboot
# load deploy/ardupilot_atlas.parm on the FC, reboot FC
python training/benchmark.py                   # confirm >= 15 fps
python -m atlas check                          # props OFF: sensors, camera+model, FC link
python -m atlas run --mission missions/site.yaml
```
Nothing arms until preflight passes **and** the pilot's MISSION-GO switch (RC7) is high.
The pilot keeps a transmitter and can take over in LOITER/STABILIZE at any time.

## Training the drop-zone model

```bash
pip install -r requirements-train.txt
python training/synth_data.py --out data/synth --n 6000            # bootstrap set
python training/yolo_to_csv.py --root data/field                   # real frames from the vehicle camera
python training/train.py --data data/synth data/field --out models
python training/quantize.py --model models/dropzone_float.keras --out models/dropzone_mnv2_int8.tflite
```
`quantize.py` fails the build if INT8 accuracy drops more than 2 points below float.

`models/dropzone_mnv2_int8_synthetic.tflite` is a smoke-test model trained from scratch on
synthetic images only. It proves the pipeline end to end; train on real frames from the vehicle
camera (starting from ImageNet weights) before flying.

## Results in simulation

- 200 randomised missions (`sim.monte_carlo --runs 200 --seed 1000`): 197 delivered, 3 safe aborts,
  1 light contact in gusty 4.9 m/s wind; release error median 0.38 m, 95th percentile 1.07 m.
- `docs/sim_demo.png`: the demo mission, top view and altitude by phase.
- `docs/monte_carlo_200.json`: every run of the sweep.
