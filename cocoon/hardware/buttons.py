import RPi.GPIO as GPIO
import threading
import logging
import time
import config

logger = logging.getLogger(__name__)


class ButtonController:
    """
    ButtonController using polling instead of GPIO edge detection.

    START button:
    - Starts the main program when pressed.
    - Disabled while the program is running.

    STOP button:
    - Stops the running program.
    - Does nothing if no program is running.
    """

    def __init__(self, start_pin=config.START_BUTTON_PIN, stop_pin=config.STOP_BUTTON_PIN):
        self.start_pin = start_pin
        self.stop_pin = stop_pin

        # Threading events
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.running_event = threading.Event()

        GPIO.setmode(GPIO.BCM)

        # Setup pins
        GPIO.setup(self.start_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.stop_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # For detecting state changes (debounce)
        self.last_start_state = GPIO.input(self.start_pin)
        self.last_stop_state = GPIO.input(self.stop_pin)

        # Start polling thread
        self._polling_thread = threading.Thread(target=self._poll_buttons, daemon=True)
        self._polling_thread.start()

        logger.info("Buttons initialized using polling (no edge detection).")

    # ---------------- INTERNAL THREAD ---------------- #

    def _poll_buttons(self):
        """
        Continuously checks button states and detects presses.
        """
        while True:
            start_state = GPIO.input(self.start_pin)
            stop_state = GPIO.input(self.stop_pin)

            # Detect START press (HIGH -> LOW)
            if self.last_start_state == GPIO.HIGH and start_state == GPIO.LOW:
                self._start_pressed()

            # Detect STOP press (HIGH -> LOW)
            if self.last_stop_state == GPIO.HIGH and stop_state == GPIO.LOW:
                self._stop_pressed()

            self.last_start_state = start_state
            self.last_stop_state = stop_state

            time.sleep(0.05)  # debounce delay

    # ---------------- CALLBACK LOGIC ---------------- #

    def _start_pressed(self):
        if self.running_event.is_set():
            logger.warning("START button pressed but system already running (ignored).")
            return

        logger.info("START button pressed.")
        self.start_event.set()

    def _stop_pressed(self):
        if not self.running_event.is_set():
            logger.warning("STOP button pressed but no script is running.")
            return

        logger.warning("FORCE STOP button pressed!")
        self.stop_event.set()

    # ---------------- CONTROL METHODS ---------------- #

    def wait_for_start(self):
        logger.info("Waiting for START button...")
        self.start_event.wait()
        logger.info("Start signal received.")

    def set_running(self, state: bool):
        if state:
            self.running_event.set()
            logger.info("System state: RUNNING")
        else:
            self.running_event.clear()
            logger.info("System state: IDLE")

    def should_stop(self):
        return self.stop_event.is_set()

    def reset(self):
        self.start_event.clear()
        self.stop_event.clear()
        self.running_event.clear()
        logger.info("Button states reset.")

    def cleanup(self):
        GPIO.cleanup([self.start_pin, self.stop_pin])
        logger.info("Button GPIO cleaned up.")