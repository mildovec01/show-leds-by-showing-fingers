#!/usr/bin/env python3
import time, numpy as np, cv2
from picamera2 import Picamera2
from gpiozero import LED

# ===== KONFIG =====
LED_PINS = [17, 27, 22]     # 1–3 LED
RES = (960, 540)
FLIP = True

# HSV detekce "černé" (nízký jas, nízká saturace)
LOW = np.array([0,   0,   0],    dtype=np.uint8)   # H, S, V min
HIGH= np.array([179, 70,  70],   dtype=np.uint8)   # H max, S max, V max (V ~70/255 je tmavé)

MIN_BLOB_AREA = 200          # min. plocha jedné černé tečky (px)
DWELL_MS = 250               # potvrzení změny počtu
MAX_FINGERS = 3              # 3 LED
PRINT_CHANGES = True

# ===== GPIO =====
leds = [LED(p) for p in LED_PINS]
def show_count(n):
    for i, L in enumerate(leds, start=1):
        (L.on() if i <= n else L.off())

# ===== Kamera =====
p = Picamera2()
p.configure(p.create_preview_configuration(main={"size": RES, "format": "RGB888"}))
p.start(); time.sleep(0.2)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
stable = 0
last_cand = 0
since = time.monotonic()
show_count(stable)
if PRINT_CHANGES: print(f"LEDs: {stable}/{MAX_FINGERS} (start)")

try:
    while True:
        frame = p.capture_array()
        if FLIP:
            frame = cv2.flip(frame, 1)

        # Černá = nízká hodnota V a nízká saturace S
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, LOW, HIGH)

        # Omezíme se jen na horní část obrazu (nahoře máš prsty)
        H, W = mask.shape
        roi = mask[:int(H * 0.7), :]  # zohledníme jen 70 % shora
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
        roi = cv2.medianBlur(roi, 5)

        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in cnts if cv2.contourArea(c) >= MIN_BLOB_AREA]

        fingers = min(len(blobs), MAX_FINGERS)

        # Stabilizace: dwell
        now = time.monotonic()
        if fingers != last_cand:
            last_cand = fingers
            since = now
        else:
            if (now - since) * 1000 >= DWELL_MS and stable != last_cand:
                stable = last_cand
                show_count(stable)
                if PRINT_CHANGES:
                    print(f"LEDs: {stable}/{MAX_FINGERS}")

        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    show_count(0)
    p.stop()
    print("Bye.")

