import time
from collections import deque

import numpy as np
from gpiozero import LED
from picamera2 import Picamera2
import mediapipe as mp
import cv2  # jen kvůli flipu, nic nezobrazuju

# --- TUNABLES ---
DEBUG = True
FLIP = True                 # zrcadlení (při self-view je to přirozenější)
RES = (960, 540)            # víc detailu než 640x480
DET_CONF = 0.5              # snížené prahy ať se chytne snáz
TRK_CONF = 0.5
WINDOW = 5                  # kratší okno
VOTE_NEED = 3               # vote >= 3/5

# --- GPIO (BCM) ---
led_pins = [17, 27, 22]  # 1,2,3 prsty -> uprav podle sebe
leds = [LED(p) for p in led_pins]

def show_count(n: int):
    for i, led in enumerate(leds, start=1):
        if i <= n:
            led.on()
        else:
            led.off()

def count_fingers(hand_landmarks, handedness_label: str) -> int:
    lm = hand_landmarks.landmark
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    fingers = 0
    for tip, pip in zip(tips, pips):
        if lm[tip].y < lm[pip].y:
            fingers += 1

    thumb_tip = lm[4]; thumb_mcp = lm[2]
    if handedness_label == "Right":
        if thumb_tip.x < thumb_mcp.x:
            fingers += 1
    else:
        if thumb_tip.x > thumb_mcp.x:
            fingers += 1
    return fingers

# --- Kamera ---
picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"size": RES, "format": "RGB888"}))
picam.start()
time.sleep(0.2)

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=DET_CONF,
                       min_tracking_confidence=TRK_CONF)

history = deque(maxlen=WINDOW)
stable_count = -1

last_debug = 0.0
print("Running headless finger counter. Ctrl+C to stop.")
try:
    while True:
        frame = picam.capture_array()   # RGB
        if FLIP:
            frame = cv2.flip(frame, 1)  # horizontální zrcadlení

        results = hands.process(frame)

        raw_count = 0
        label = "Right"
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            if results.multi_handedness:
                label = results.multi_handedness[0].classification[0].label
            raw_count = count_fingers(lms, label)

        # mapuj do 0–3 (máš tři LED)
        c = max(0, min(3, raw_count))
        history.append(c)

        # hlasování proti blikání
        # najdi mód (nejčastější hodnotu)
        vals = list(history)
        candidate = max(set(vals), key=vals.count)
        if vals.count(candidate) >= VOTE_NEED and candidate != stable_count:
            stable_count = candidate
            show_count(stable_count)
            print(f"LEDs: {stable_count}/3")

        # 1× za vteřinu vypiš debug (ať víš, jestli ruku vůbec vidí)
        now = time.time()
        if DEBUG and now - last_debug >= 1.0:
            last_debug = now
            hands_seen = 1 if results.multi_hand_landmarks else 0
            print(f"[DBG] hand={hands_seen} raw={raw_count} voted={candidate} stable={stable_count} hist={list(history)}")

        time.sleep(0.005)

except KeyboardInterrupt:
    pass
finally:
    show_count(0)
    hands.close()
    picam.stop()
    print("Bye.")
