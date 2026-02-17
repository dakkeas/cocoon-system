import RPi.GPIO as GPIO
import threading
import logging
import time
import config

logger = logging.getLogger(__name__)


class ButtonController:
    def __init__(self, start_pin=config.START_BUTTON_PIN, stop_pin=config.STOP_BUTTON_PIN):
        self.start_pin = start_pin
        self.stop_pin = stop_pin

        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.running_event = threading.Event()  # True when main loop is running

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.start_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.stop_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        GPIO.add_event_detect(
            self.start_pin,
            GPIO.FALLING,
            callback=self._start_pressed,
            bouncetime=300
        )

        GPIO.add_event_detect(
            self.stop_pin,
            GPIO.FALLING,
            callback=self._stop_pressed,
            bouncetime=300
        )

        logger.info(
            "Buttons initialized (START: GPIO %d, STOP: GPIO %d)",
            self.start_pin, self.stop_pin
        )

    # ---------------- CALLBACKS ---------------- #

    def _start_pressed(self, channel):
        if self.running_event.is_set():
            logger.warning("START button pressed but system already running (ignored).")
            return

        logger.info("START button pressed.")
        self.start_event.set()

    def _stop_pressed(self, channel):
        logger.warning("FORCE STOP button pressed!")
        self.stop_event.set()

    # ---------------- CONTROL METHODS ---------------- #

    def wait_for_start(self):
        logger.info("Waiting for START button...")
        self.start_event.wait()
        logger.info("Start signal received.")

    def set_running(self, state: bool):
        """
        Call True when main loop starts, False when it stops
        """
        if state:
            self.running_event.set()
            logger.info("System state: RUNNING")
        else:
            self.running_event.clear()
            logger.info("System state: IDLE")

    def should_stop(self):
        return self.stop_event.is_set()

    def reset(self):
        """
        Reset after force stop
        """
        self.start_event.clear()
        self.stop_event.clear()
        self.running_event.clear()
        logger.info("Button states reset.")

    def cleanup(self):
        GPIO.cleanup([self.start_pin, self.stop_pin])
        logger.info("Button GPIO cleaned up.")
