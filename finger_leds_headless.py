#!/usr/bin/env python3
import time, math
from collections import deque
import numpy as np
import cv2
from gpiozero import LED
from picamera2 import Picamera2

# ===== KONFIG =====
LED_PINS = [17, 27, 22]       # 1–3 prsty
RES = (960, 540)
FLIP = True
MIN_AREA = 9000               # min. plocha ruky (px)
MAX_MOVE = 35                 # max. posun středu ROI mezi snímky (px) – když víc, neaktualizujeme (stabilita)
DEPTH_MIN = 1200              # min. hloubka defektu (ostrá „V“ mezi prsty)
ANGLE_MAX = 80                # max. úhel ve far-point (ostřejší = prst)
MEDIAN_WIN = 7                # okno na medián (liché)
DWELL_MS = 300                # nová hodnota musí držet
PRINT_CHANGES = True

# ===== GPIO =====
leds = [LED(p) for p in LED_PINS]
def show_count(n: int):
    for i, L in enumerate(leds, start=1):
        (L.on() if i <= n else L.off())

# ===== Pomocné =====
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
LOWER1 = np.array([0, 25, 40]);  UPPER1 = np.array([20, 200, 255])
LOWER2 = np.array([160,25, 40]); UPPER2 = np.array([179,200, 255])

def skin_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, LOWER1, UPPER1) | cv2.inRange(hsv, LOWER2, UPPER2)
    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    return m

def biggest_contour(mask):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA: return None
    return c

def count_fingers(cnt):
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3: return 0
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None: return 0

    fingers = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        a = np.linalg.norm(cnt[e][0]-cnt[s][0])
        b = np.linalg.norm(cnt[f][0]-cnt[s][0])
        c = np.linalg.norm(cnt[e][0]-cnt[f][0])
        if b*c == 0: 
            continue
        cosA = (b*b + c*c - a*a) / (2*b*c)
        cosA = np.clip(cosA, -1.0, 1.0)
        angle = math.degrees(math.acos(cosA))
        if angle < ANGLE_MAX and d > DEPTH_MIN:
            fingers += 1
    # defekty ~ mezery -> prstů ~ defekty+1, omezíme na 0–3
    return max(0, min(3, fingers + 1))

def centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0: return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

# ===== Kamera =====
p = Picamera2()
p.configure(p.create_preview_configuration(main={"size": RES, "format": "RGB888"}))
p.start(); time.sleep(0.2)

median_buf = deque(maxlen=MEDIAN_WIN)
stable = 0
last_output = 0
last_change_t = time.monotonic()
last_centroid = None

show_count(stable)
if PRINT_CHANGES: print(f"LEDs: {stable}/3 (start)")

try:
    while True:
        frame = p.capture_array()
        if FLIP: frame = cv2.flip(frame, 1)

        mask = skin_mask(frame)
        cnt = biggest_contour(mask)
        if cnt is None:
            # žádná ruka -> držíme, jen resetujeme median buffer (ať nenosí starý mód)
            median_buf.clear()
            time.sleep(0.01)
            continue

        # gate: ruka se nesmí moc hýbat (omezíme na malé posuny)
        c = centroid(cnt)
        if c and last_centroid:
            if abs(c[0]-last_centroid[0]) > MAX_MOVE or abs(c[1]-last_centroid[1]) > MAX_MOVE:
                # velký pohyb -> neaktualizuj (drž poslední stabilní)
                time.sleep(0.01)
                continue
        if c: last_centroid = c

        # spočítej prsty pro aktuální snímek
        curr = count_fingers(cnt)
        median_buf.append(curr)

        # robustní kandidát: medián z posledních N
        med = int(np.median(median_buf)) if median_buf else stable

        # dwell + hysteréze
        now = time.monotonic()
        if med != last_output:
            last_output = med
            last_change_t = now
        else:
            if (now - last_change_t) * 1000 >= DWELL_MS and stable != last_output:
                stable = last_output
                show_count(stable)
                if PRINT_CHANGES:
                    print(f"LEDs: {stable}/3")

        time.sleep(0.005)

except KeyboardInterrupt:
    pass
finally:
    show_count(0)
    p.stop()
    print("Bye.")
