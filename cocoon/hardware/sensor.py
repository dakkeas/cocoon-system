import RPi.GPIO as GPIO
import time
import config


class IR_Sensor:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(config.SENSOR_PIN, GPIO.IN)
        GPIO.setup(config.LED_PIN, GPIO.OUT)

    def read(self):
        """
        Returns 1 if white surface detected, 0 if black.
        Also controls LED automatically.
        """
        sensor_value = GPIO.input(config.SENSOR_PIN)

        if sensor_value == GPIO.HIGH:
            GPIO.output(config.LED_PIN, GPIO.HIGH) # turn on led
            return 1 # white
        else:
            GPIO.output(config.LED_PIN, GPIO.LOW) # turn off led
            return 0 #black

    def cleanup(self):
        GPIO.cleanup()
