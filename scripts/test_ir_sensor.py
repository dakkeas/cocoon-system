import RPi.GPIO as GPIO
import time
# import confikg


GPIO.setmode(GPIO.BCM)
GPIO.setup(1, GPIO.IN)
        # GPIO.setup(config.LED_PIN, GPIO.OUT)

"""
Returns 1 if white surface detected, 0 if black.
Also controls LED automatically.
"""
while True:
    sensor_value = GPIO.input(1)

    if sensor_value == GPIO.HIGH: # black
    # GPIO.output(config.LED_PIN, GPIO.HIGH) # turn on led
        print(f"{sensor_value} : BLACK SURFACE DETECTED")
        # return 1 # white
    else:
        print(f"{sensor_value} : WHITE SURFACE DETECTED")
    # GPIO.output(config.LED_PIN, GPIO.LOW) # turn off led
    time.sleep(0.5)
        # return 0 #black

