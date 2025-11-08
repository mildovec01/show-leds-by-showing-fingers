#!/usr/bin/env python3
import time, re, os, sys
import serial
from gpiozero import LED

# ===== KONFIG =====
PORT = "/dev/ttyACM0"      # pokud máš jiný, změň
BAUD = 115200
DWELL_MS = 200
LED_PINS = [17, 27, 22]    # 1–3 LED (BCM)

# ===== GPIO =====
leds = [LED(p) for p in LED_PINS]
def show(n):
    for i, L in enumerate(leds, start=1):
        (L.on() if i <= n else L.off())

def main():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        print(f"[INFO] Připojeno k {PORT} @ {BAUD} bps")
    except Exception as e:
        print(f"[ERR] Nelze otevřít {PORT}: {e}")
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
            if not chunk:
                time.sleep(0.01)
                continue

            buf += chunk
            if b"\n" not in buf:
                continue

            lines = buf.split(b"\n")
            buf = lines[-1]
            for raw in lines[:-1]:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                # hledáme "Num:X"
                m = re.search(r'Num[:=]\s*(\d)', line, re.IGNORECASE)
                if not m:
                    continue
                d = int(m.group(1))
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

    except KeyboardInterrupt:
        pass
    finally:
        show(0)
        try: ser.close()
        except: pass
        print("Bye.")

if __name__ == "__main__":
    main()


