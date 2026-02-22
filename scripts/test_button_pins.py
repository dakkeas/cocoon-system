import RPi.GPIO as GPIO
import time

START_BUTTON_PIN = 4
STOP_BUTTON_PIN = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(START_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(STOP_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Button test started. Press buttons...")

last_start_state = GPIO.input(START_BUTTON_PIN)
last_stop_state = GPIO.input(STOP_BUTTON_PIN)

try:
    while True:
        start_state = GPIO.input(START_BUTTON_PIN)
        stop_state = GPIO.input(STOP_BUTTON_PIN)

        # Detect falling edge (button press)
        if last_start_state == GPIO.HIGH and start_state == GPIO.LOW:
            print("START button pressed")

        if last_stop_state == GPIO.HIGH and stop_state == GPIO.LOW:
            print("STOP button pressed")

        last_start_state = start_state
        last_stop_state = stop_state

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    GPIO.cleanup()