#!/usr/bin/env python3
import sys, time, cv2, numpy as np
from gpiozero import LED
import mediapipe as mp

# ===== KONFIG =====
LED_PINS = [17, 27, 22]    # 1–3 LED (BCM)
DWELL_MS = 300             # potvrzení změny (stabilizace)
MAX_FINGERS = 3            # kolik LED reálně máme
FLIP = True                # zrcadlení (když by palec zlobil, dej False)
DET_CONF = 0.5
TRK_CONF = 0.5

# ===== GPIO =====
leds = [LED(p) for p in LED_PINS]
def show(n):
    for i, L in enumerate(leds, start=1):
        (L.on() if i <= n else L.off())

def clamp(v,a,b): return a if v<a else (b if v>b else v)

# ===== počítání prstů z landmarků =====
def count_fingers_from_landmarks(lms, handed_label: str) -> int:
    lm = lms.landmark
    tips = [8,12,16,20]; pips = [6,10,14,18]
    fingers = sum(1 for tip,pip in zip(tips,pips) if lm[tip].y < lm[pip].y)
    # palec podle orientace ruky
    thumb_tip = lm[4]; thumb_mcp = lm[2]
    if handed_label == "Right":
        if thumb_tip.x < thumb_mcp.x: fingers += 1
    else:
        if thumb_tip.x > thumb_mcp.x: fingers += 1
    return fingers

# ===== MediaPipe Hands =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=DET_CONF,
                       min_tracking_confidence=TRK_CONF)

# ===== MJPEG parser pro rpicam-vid -> stdin =====
SOI = b"\xff\xd8"  # JPEG start
EOI = b"\xff\xd9"  # JPEG end
buf = bytearray()

# stabilizace (dwell + hysteréze)
stable = 0
last_cand = 0
since = time.monotonic()
show(stable)
print(f"LEDs: {stable}/{MAX_FINGERS} (start)")

def process_frame(rgb):
    global stable, last_cand, since
    if FLIP:
        rgb = cv2.flip(rgb, 1)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        lms = res.multi_hand_landmarks[0]
        handed = "Right"
        if res.multi_handedness:
            handed = res.multi_handedness[0].classification[0].label
        raw = count_fingers_from_landmarks(lms, handed)
        cand = clamp(raw, 0, MAX_FINGERS)
    else:
        cand = stable  # když ruka zmizí, drž poslední stav

    now = time.monotonic()
    if cand != last_cand:
        last_cand = cand; since = now
    else:
        if (now - since)*1000 >= DWELL_MS and stable != last_cand:
            stable = last_cand
            show(stable)
            print(f"LEDs: {stable}/{MAX_FINGERS}")

try:
    while True:
        chunk = sys.stdin.buffer.read(4096)
        if not chunk:
            time.sleep(0.001)
            continue
        buf.extend(chunk)
        # parsuj postupně celé JPEG snímky
        while True:
            start = buf.find(SOI)
            if start < 0:
                if len(buf) > 1024*1024: buf.clear()
                break
            end = buf.find(EOI, start+2)
            if end < 0:
                break
            jpg = bytes(buf[start:end+2])
            del buf[:end+2]
            bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None: 
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            process_frame(rgb)
except KeyboardInterrupt:
    pass
finally:
    show(0)
    hands.close()
    print("Bye.")
