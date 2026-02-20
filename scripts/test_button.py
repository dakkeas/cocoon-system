

import RPi.GPIO as GPIO
import time
import os
import config

BUTTON_PIN = config.BUTTON_PIN

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Waiting for button press...")

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:  # button pressed
            print("Button pressed!")
            # os.system("python3 your_script.py")  # run your script
            time.sleep(1)  # prevent multiple triggers
        time.sleep(0.1)

except KeyboardInterrupt:
    GPIO.cleanup()

