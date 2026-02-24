
import threading
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class IRMotorController(threading.Thread):
    """
    Controls motor using IR sensor.
    Runs motor while IR sensor reads the target color.
    """

    def __init__(self, motor_system, ir_sensor, target_color, speed=70):
        super().__init__(daemon=True)
        self.motor = motor_system
        self.sensor = ir_sensor
        self.target_color = target_color
        self.speed = speed
        self.stop_event = threading.Event()
        self.finished = threading.Event()

    def run(self):
        logger.info(f"IRMotorController started (target_color={self.target_color})")
        motor_running = False
        try:
            while not self.stop_event.is_set():
                reading = self.sensor.read()
                # print(f'SENSOR READING ---------------> {reading}')
                # print(f'SENSOR READING ---------------> {"WHITE" if reading == 0 else "BLACK"}')
                # print(f'--------------------> RUN IF {self.target_color} DETECTED')
                if reading == self.target_color:
                    if not motor_running:
                        self.motor.start(self.speed)
                        motor_running = True
                else:
                    if motor_running:
                        self.motor.stop()
                        motor_running = False
                        self.finished.set()
                        break
                time.sleep(0.01)  # small delay to be CPU-friendly
        finally:
            self.motor.stop()
            self.finished.set()
            logger.info("IRMotorController finished safely.")

    def wait_until_finished(self):
        self.finished.wait()