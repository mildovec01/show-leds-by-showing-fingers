import time
from collections import deque
import numpy as np
import cv2
from gpiozero import LED
from picamera2 import Picamera2

# ==== NASTAVENÍ ====
LED_PINS = [17, 27, 22]      # 1–3 prsty -> 3 LED
RES = (960, 540)             # víc detailu pomáhá
FLIP = True                  # zrcadlení (při self-view přirozené)
WINDOW = 5                   # stabilizace
VOTE_NEED = 3                # majorita z okna
DEBUG_EVERY = 1.0            # sekundy (0 = nevypisovat)

# HSV prahování kůže (užitečné start hodnoty; můžeš doladit)
# Pozn.: světlejší/teplejší světlo -> tyto rozsahy obvykle fungují.
LOWER1 = np.array([0, 30, 60])
UPPER1 = np.array([20, 180, 255])
LOWER2 = np.array([160, 30, 60])
UPPER2 = np.array([179, 180, 255])

# Morfologie
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# ==== GPIO ====
leds = [LED(p) for p in LED_PINS]
def show_count(n: int):
    for i, led in enumerate(leds, start=1):
        if i <= n: led.on()
        else: led.off()

def count_fingers_from_mask(mask: np.ndarray) -> int:
    # Najdi největší konturu = ruka
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 3000:  # příliš malé (nic/smetí)
        return 0

    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0

    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return 0

    # Počítej "mezery" mezi prsty (defekty) s rozumnou geometrií
    fingers = 0
    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        start = cnt[s][0]; end = cnt[e][0]; far = cnt[f][0]
        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)
        # kosinová věta na úhel ve "far" bodě
        if b * c == 0: 
            continue
        cosA = (b*b + c*c - a*a) / (2*b*c)
        cosA = np.clip(cosA, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosA))

        # Heuristiky:
        # - špička mezi prsty má malý úhel (ostrá "V"), typicky < 80°
        # - depth (v 1/256 px) musí být dost velký => reálná mezera
        if angle < 80 and depth > 1200:
            fingers += 1

    # Defekty ~ počet mezer, takže prstů je defekty + 1
    fingers = min(5, fingers + 1)

    # Mapuj na 0–3 LED
    return min(3, max(0, fingers))

def preprocess(frame: np.ndarray) -> np.ndarray:
    # HSV + dvě pásma červené (kvůli odstínům pleti), OR, morfologie
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, LOWER1, UPPER1)
    mask2 = cv2.inRange(hsv, LOWER2, UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    return mask

# ==== Kamera ====
picam = Picamera2()
picam.configure(picam.create_preview_configuration(main={"size": RES, "format":"RGB888"}))
picam.start()
time.sleep(0.2)

history = deque(maxlen=WINDOW)
stable = -1
last_dbg = 0.0

print("Running no-mediapipe finger counter. Ctrl+C to stop.")
try:
    while True:
        frame = picam.capture_array()  # RGB
        if FLIP:
            frame = cv2.flip(frame, 1)

        mask = preprocess(frame)
        c = count_fingers_from_mask(mask)

        history.append(c)
        vals = list(history)
        candidate = max(set(vals), key=vals.count)
        if vals.count(candidate) >= VOTE_NEED and candidate != stable:
            stable = candidate
            show_count(stable)
            print(f"LEDs: {stable}/3")

        now = time.time()
        if DEBUG_EVERY > 0 and now - last_dbg >= DEBUG_EVERY:
            last_dbg = now
            print(f"[DBG] c={c} voted={candidate} stable={stable} hist={list(history)}")

        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    show_count(0)
    picam.stop()
    print("Bye.")

