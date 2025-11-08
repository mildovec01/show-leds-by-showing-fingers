from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time

# --- Nastavení LED ---
LED1 = 17
LED2 = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED1, GPIO.OUT)
GPIO.setup(LED2, GPIO.OUT)

# --- Nastavení kamery IMX500 ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(ai={"model": "/usr/share/imx500/models/digits.onnx"})
picam2.configure(config)
picam2.start()

print("Kamera běží, čekám na čísla 1 nebo 2...")

try:
    while True:
        result = picam2.ai_results.get()
        if not result:
            continue

        # Výsledek AI je např. {'digit': 1}
        digit = result.get("digit", None)
        if digit == 1:
            GPIO.output(LED1, True)
            GPIO.output(LED2, False)
            print("Rozpoznáno číslo 1 → LED1 svítí")
        elif digit == 2:
            GPIO.output(LED1, True)
            GPIO.output(LED2, True)
            print("Rozpoznáno číslo 2 → LED1 a LED2 svítí")
        else:
            GPIO.output(LED1, False)
            GPIO.output(LED2, False)

        time.sleep(0.1)

except KeyboardInterrupt:
    pass

finally:
    GPIO.cleanup()
    picam2.stop()


