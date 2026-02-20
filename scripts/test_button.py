
import RPi.GPIO as GPIO
import time
import os
import config

BUTTON_PIN = config.START_BUTTON_PIN

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

button_already_pressed = False

def button_pressed(channel):
    global button_already_pressed
    
    if GPIO.input(BUTTON_PIN) == GPIO.LOW and not button_already_pressed:
        print("Button clicked!")
        # os.system("python3 your_script.py")
        button_already_pressed = True

    elif GPIO.input(BUTTON_PIN) == GPIO.HIGH:
        # reset when toggle is switched back off
        button_already_pressed = False

GPIO.add_event_detect(BUTTON_PIN, GPIO.BOTH, callback=button_pressed, bouncetime=300)

print("Ready. Toggle the button.")

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()