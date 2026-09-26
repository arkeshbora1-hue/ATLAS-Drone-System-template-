#!/usr/bin/env bash
# One-time Raspberry Pi 4 setup for ATLAS (Raspberry Pi OS Bookworm 64-bit).
# Run as the default user with sudo rights:  bash deploy/setup_pi.sh
set -euo pipefail
ATLAS_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "== system packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-numpy python3-opencv python3-picamera2 \
                        pigpio python3-pigpio i2c-tools

echo "== interfaces: I2C on, serial console off, UART hardware on"
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_serial_cons 1
sudo raspi-config nonint do_serial_hw 0
CFG=/boot/firmware/config.txt
# give the full PL011 UART to GPIO14/15 (Bluetooth moves off it) and run I2C at 400 kHz
grep -q "^dtoverlay=disable-bt" $CFG || echo "dtoverlay=disable-bt" | sudo tee -a $CFG
grep -q "^dtparam=i2c_arm_baudrate" $CFG || echo "dtparam=i2c_arm_baudrate=400000" | sudo tee -a $CFG
sudo systemctl disable --now hciuart || true

echo "== pigpio daemon (microsecond echo timing for HC-SR04)"
sudo systemctl enable --now pigpiod

echo "== python environment"
python3 -m venv --system-site-packages "$ATLAS_DIR/.venv"
"$ATLAS_DIR/.venv/bin/pip" install --upgrade pip
"$ATLAS_DIR/.venv/bin/pip" install -r "$ATLAS_DIR/requirements-pi.txt"

echo "== service"
sed "s#@ATLAS_DIR@#$ATLAS_DIR#g; s#@USER@#$USER#g" "$ATLAS_DIR/deploy/atlas.service" | sudo tee /etc/systemd/system/atlas.service >/dev/null
sudo systemctl daemon-reload
echo "Installed. The service is NOT enabled automatically."
echo "Bench check (props off):  $ATLAS_DIR/.venv/bin/python -m atlas check"
echo "Enable for field use:     sudo systemctl enable --now atlas"
echo "Reboot now to apply UART/I2C changes."
