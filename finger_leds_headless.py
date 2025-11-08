#!/usr/bin/env python3
import time
from gpiozero import LED
from picamera2 import Picamera2
import mediapipe as mp
import cv2  # jen kvůli flipu, nic nezobrazuju

# ===== KONFIG =====
LED_PINS = [17, 27, 22]   # 1–3 prsty -> 3 LED
RES = (960, 540)          # kompromis rychlost/kvalita
FLIP = True               # zrcadlení (self-view)
DWELL_MS = 300            # změnu potvrdím až když drží >= 300 ms
MAX_FINGERS = 3           # kolik LED reálně máme (ořízne 4–5 na 3)
DET_CONF = 0.5            # prahy detekce/tracking
TRK_CONF = 0.5

# ===== GPIO =====
leds = [LED(p) for p in LED_PINS]
def show_count(n: int):
    for i, led in enumerate(leds, start=1):
        (led.on() if i <= n else led.off())

# ===== MediaPipe helpers =====
def count_fingers_from_landmarks(lms, handed_label: str) -> int:
    """
    5-prstová logika (MediaPipe Hands):
      - ukazováček..malík: tip.y < pip.y
      - palec: Right => tip.x < mcp.x, Left => tip.x > mcp.x
    """
    lm = lms.landmark  # DŮLEŽITÉ: přístup přes .landmark
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers = 0
    for tip, pip in zip(tips, pips):
        if lm[tip].y < lm[pip].y:
            fingers += 1
    # palec
    thumb_tip = lm[4]; thumb_mcp = lm[2]
    if handed_label == "Right":
        if thumb_tip.x < thumb_mcp.x:
            fingers += 1
    else:
        if thumb_tip.x > thumb_mcp.x:
            fingers += 1
    return fingers

def clamp(v, a, b): return a if v < a else (b if v > b else v)

# ===== Kamera =====
picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"size": RES, "format": "RGB888"}))
picam.start()
time.sleep(0.2)

# ===== MediaPipe =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=DET_CONF,
                       min_tracking_confidence=TRK_CONF)

# ===== Stabilizace: dwell + hysteréze =====
stable = 0
last_candidate = 0
since = time.monotonic()
show_count(stable)
print(f"LEDs: {stable}/{MAX_FINGERS} (start)")

print("Running headless. Ctrl+C to stop.")
try:
    while True:
        frame = picam.capture_array()  # RGB
        if FLIP:
            frame = cv2.flip(frame, 1)

        res = hands.process(frame)

        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks[0]
            label = "Right"
            if res.multi_handedness:
                label = res.multi_handedness[0].classification[0].label

            count_0_5 = count_fingers_from_landmarks(lms, label)
            candidate = clamp(count_0_5, 0, MAX_FINGERS)
        else:
            # žádná ruka -> držíme poslední stabilní stav (neblikáme)
            candidate = stable

        now = time.monotonic()
        if candidate != last_candidate:
            last_candidate = candidate
            since = now
        else:
            if (now - since) * 1000 >= DWELL_MS and stable != last_candidate:
                stable = last_candidate
                show_count(stable)
                print(f"LEDs: {stable}/{MAX_FINGERS}")

        time.sleep(0.005)

except KeyboardInterrupt:
    pass
finally:
    show_count(0)
    hands.close()
    picam.stop()
    print("Bye.")
