#!/usr/bin/env python3
import time
from collections import deque
import math

import numpy as np
import cv2  # jen pro flip/ROI operace, žádná okna!
from gpiozero import LED
from picamera2 import Picamera2

# ==== KONFIG ====
LED_PINS = [17, 27, 22]   # 1–3 prsty => 3 LED
RES_MAIN = (960, 540)     # rozumný kompromis
FLIP = True               # zrcadlení (self-view)
MIN_HAND_AREA = 8000      # minimální plocha ruky (px) – filtruje šum
ROI_MARGIN = 60           # o kolik px rozšířit ROI okolo detekce
DWELL_MS = 300            # jak dlouho musí držet nový počet prstů, aby se přepnuly LED
MAX_FINGERS = 3           # kolik LED (ořízne 4–5 prstů na 3)

# ==== GPIO ====
leds = [LED(p) for p in LED_PINS]
def show_count(n: int):
    for i, led in enumerate(leds, start=1):
        if i <= n: led.on()
        else: led.off()

# ==== MediaPipe Hands (lazy import s fallbackem) ====
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# --- pomocné ---
def clamp(v, a, b): return a if v < a else (b if v > b else v)

def biggest_skin_roi(rgb):
    """Najdi hrubě ruku přes HSV masku kůže a vrať bounding box (x,y,w,h) nebo None."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # dvě červené oblasti + menší nároky na S/V (tolerantní)
    lower1 = np.array([0, 25, 40]);  upper1 = np.array([20, 200, 255])
    lower2 = np.array([160, 25, 40]); upper2 = np.array([179, 200, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_HAND_AREA: return None
    x,y,w,h = cv2.boundingRect(cnt)
    return (x, y, w, h)

def count_fingers_from_landmarks(lm, handed_label: str) -> int:
    """
    5-prstová logika z landmarků:
    - ukazováček..malík: tip.y < pip.y
    - palec: pro Right tip.x < mcp.x, pro Left tip.x > mcp.x
    """
    tips = [8,12,16,20]
    pips = [6,10,14,18]
    fingers = 0
    for tip, pip in zip(tips, pips):
        if lm[tip].y < lm[pip].y:
            fingers += 1
    # palec
    thumb_tip = lm[4]; thumb_mcp = lm[2]
    if handed_label == "Right":
        if thumb_tip.x < thumb_mcp.x: fingers += 1
    else:
        if thumb_tip.x > thumb_mcp.x: fingers += 1
    return fingers

def safe_crop(img, x, y, w, h):
    H, W = img.shape[:2]
    x0 = clamp(x, 0, W-1); y0 = clamp(y, 0, H-1)
    x1 = clamp(x+w, 0, W); y1 = clamp(y+h, 0, H)
    if x1 <= x0 or y1 <= y0:
        return img, (0,0,W,H)
    return img[y0:y1, x0:x1], (x0,y0,x1-x0,y1-y0)

# ==== Kamera ====
picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"size": RES_MAIN, "format":"RGB888"}))
picam.start()
time.sleep(0.2)

# ==== MediaPipe init (pokud dostupný) ====
if MP_AVAILABLE:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
else:
    hands = None

# ==== Stabilizace: dwell + hysteréze ====
stable_count = 0
last_candidate = 0
candidate_since = time.monotonic()

# Držíme i poslední ROI pro rychlé vyříznutí ruky
last_roi = None   # (x,y,w,h) v souřadnicích celého snímku

print("Running. Ctrl+C to stop.")
try:
    while True:
        frame = picam.capture_array()   # RGB
        if FLIP:
            frame = cv2.flip(frame, 1)
        H,

