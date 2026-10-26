#!/usr/bin/env bash
# Build and upload the selected FOCmini firmware, then monitor it. The optional
# motor-test path asks for a separate confirmation before requesting one spin.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
pio="${PIO:-$repo_root/.venv/bin/pio}"
port="${PORT:-}"
profile="as5600"
program="foc"

usage() {
  cat <<'EOF'
Usage: hw/focmini/run_motor.sh [--port /dev/ttyUSB0]
       [--program foc|motor-test] [--profile as5600|open-loop]

Defaults to the AS5600 FOC program. The motor-test program is sensorless; after
upload it waits for READY, asks for a separate spin confirmation, and requests
one bounded 10 s open-loop run. If --port is omitted, a single USB serial port
is selected automatically; specify --port when more than one is connected.
Set PIO=/path/to/pio to use another PlatformIO.
EOF
}

while (($#)); do
  case "$1" in
    --port)
      (($# >= 2)) || { usage >&2; exit 2; }
      port="$2"
      shift 2
      ;;
    --profile)
      (($# >= 2)) || { usage >&2; exit 2; }
      profile="$2"
      shift 2
      ;;
    --program)
      (($# >= 2)) || { usage >&2; exit 2; }
      program="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$program" in
  foc)
    case "$profile" in
      as5600) environment=esp32dev_as5600 ;;
      open-loop) environment=esp32dev_open_loop ;;
      *) printf 'Unsupported profile: %s\n' "$profile" >&2; usage >&2; exit 2 ;;
    esac
    selected_source=main
    ;;
  motor-test)
    environment=esp32dev_as5600
    selected_source=motor_test
    ;;
  *) printf 'Unsupported program: %s\n' "$program" >&2; usage >&2; exit 2 ;;
esac

if [[ ! -x "$pio" ]]; then
  printf 'PlatformIO executable not found or not executable: %s\n' "$pio" >&2
  printf 'Install the project build environment as described in hw/focmini/README.md.\n' >&2
  exit 1
fi

if [[ -z "$port" ]]; then
  candidates=()
  shopt -s nullglob
  for candidate in /dev/ttyUSB* /dev/ttyACM* /dev/cu.usbserial-* /dev/cu.SLAB_USBtoUART*; do
    [[ -c "$candidate" ]] && candidates+=("$candidate")
  done
  shopt -u nullglob
  if ((${#candidates[@]} != 1)); then
    printf 'Expected exactly one USB serial device; found %d.\n' "${#candidates[@]}" >&2
    if ((${#candidates[@]} > 1)); then printf '  %s\n' "${candidates[@]}" >&2; fi
    printf 'Re-run with --port /path/to/device.\n' >&2
    exit 1
  fi
  port="${candidates[0]}"
fi

if [[ ! -c "$port" ]]; then
  printf 'Serial device does not exist or is not a character device: %s\n' "$port" >&2
  exit 1
fi

printf 'Program: %s (%s.cpp; PlatformIO env %s)\nSerial port: %s\n' \
  "$program" "$selected_source" "$environment" "$port"
printf 'Keep motor power disconnected for upload. Continue? [y/N] '
read -r answer
[[ "$answer" == [yY] || "$answer" == [yY][eE][sS] ]] || { printf 'Cancelled.\n'; exit 0; }

cd "$repo_root"
FOCMINI_PROGRAM="$selected_source" "$pio" run -d hw/focmini -e "$environment"
FOCMINI_PROGRAM="$selected_source" "$pio" run -d hw/focmini -e "$environment" \
  -t upload --upload-port "$port"

printf '\nUpload complete. Before continuing, connect the motor supply only if the\n'
printf 'wiring is verified, the supply is current-limited, the motor is unloaded,\n'
printf 'and the shaft is clear. Opening serial may reset the ESP32.\n'
if [[ "$program" == motor-test ]]; then
  printf 'The helper will watch startup and ask before requesting one 10 s run.\n'
  printf 'It will abort if READY is not reported or the DRI FAULT is active.\n'
  "$repo_root/.venv/bin/python" "$script_dir/motor_test_serial.py" --port "$port"
  exit 0
fi
printf 'Press Enter to open the monitor, or Ctrl-C to stop here.\n'
read -r
"$pio" device monitor -d hw/focmini -b 115200 --port "$port"
