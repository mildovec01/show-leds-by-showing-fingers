#!/usr/bin/env python3
import time, re, sys, os
from gpiozero import LED
import serial

# ===== KONFIG =====
SERIAL_PORTS = ["/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyAMA0"]  # zkusí postupně
BAUD = 115200          # když nefunguje, změň na 9600
DWELL_MS = 200
LED_PINS = [17, 27, 22]  # 1–3 LED (BCM)

# ===== GPIO =====
leds = [LED(p) for p in LED_PINS]
def show(n: int):
    for i, L in enumerate(leds, start=1):
        (L.on() if i <= n else L.off())

def open_serial():
    last_err = None
    for dev in SERIAL_PORTS:
        try:
            if os.path.exists(dev):
                return serial.Serial(dev, BAUD, timeout=0.1)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise RuntimeError("Nenalezen sériový port. Zkontroluj USB/UART připojení.")

def extract_digit(line: str):
    m = re.search(r'(\d)', line)
    return int(m.group(1)) if m else None

def main():
    try:
        ser = open_serial()
        print(f"[INFO] Připojeno k {ser.port} @ {BAUD} bps")
    except Exception as e:
        print(f"[ERR] Port neotevřen: {e}")
        sys.exit(1)

    stable = 0
    last_cand = 0
    since = time.monotonic()
    show(stable)
    print(f"LEDs: {stable}/3 (start)")

    buf = b""
    try:
        while True:
            chunk = ser.read(128)
            if chunk:
                buf += chunk
                if b"\n" in buf:
                    lines = buf.split(b"\n")
                    buf = lines[-1]
                    for raw in lines[:-1]:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        d = extract_digit(line)  # 0..9 nebo None
                        if d is None:
                            continue
                        candidate = d if d in (1,2,3) else 0

                        now = time.monotonic()
                        if candidate != last_cand:
                            last_cand = candidate
                            since = now
                        else:
                            if (now - since)*1000 >= DWELL_MS and stable != last_cand:
                                stable = last_cand
                                show(stable)
                                print(f"LEDs: {stable}/3")
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        show(0)
        try: ser.close()
        except: pass
        print("Bye.")

if __name__ == "__main__":
    main()

