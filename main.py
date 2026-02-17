import threading
import time
import logging
import RPi.GPIO as GPIO

from .cocoon.hardware import buttons
from .cocoon.hardware import motor
from .cocoon.hardware import sensor
from .cocoon.hardware import servo
from .cocoon import inference




# ---------------- GPIO SAFETY ---------------- #
GPIO.setwarnings(False)



# ---------------- LOGGING CONFIG ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ---------------- MOTOR + SENSOR THREAD ---------------- #
def sensor_motor_control(sensor, motors, trigger_color, stop_event, button):
    """
    Runs motor until sensor detects opposite color OR force stop is pressed.
    """
    logger.info("Motor thread started (trigger_color=%s)", trigger_color)
    motors.start()

    while not stop_event.is_set():

        # Emergency stop check
        if button.should_stop():
            logger.warning("Emergency stop detected in motor thread.")
            motors.stop()
            stop_event.set()
            return

        detected_color = sensor.read_color()
        logger.debug("Detected color: %s", detected_color)

        if detected_color != trigger_color:
            motors.stop()
            logger.info("Color changed to %s. Motors stopped.", detected_color)
            stop_event.set()
            return

        time.sleep(0.01)  # CPU-safe polling delay


vision_system = inference.VisionSystem()
motor_system = motor.MotorSystem()
ir_sensor = sensor.IR_Sensor()
servo_controller = servo.ServoController()

# ---------------- MAIN LOOP LOGIC ---------------- #
def run_main_loop(button):

    # 1. Check all systems
    # if not all([
    #     inference.is_working(),
    #     motors.is_working(),
    #     sensor.is_working(),
    #     servo.is_working()
    # ]):
    #     logger.error("System check failed. Aborting run.")
    #     return

    logger.info("All systems operational.")

    # 2. Set position forward
    motor_system.set_position("FORWARD")

    logger.info("Position set to FORWARD")

    # 3. Run inference
    results = inference.run_inference()
    logger.info("Inference completed.")

    trigger_color = 1 # 1 = white ; 0 - black/none

    # 3. 12-cycle loop
    for i in range(12):
        logger.info("Cycle %d / 12 started", i + 1)

        if button.should_stop():
            logger.warning("Force stop detected. Breaking main loop.")
            break

        servo_array = results[i]

        # --- SERVO RUNS FIRST ---
        logger.info("Starting servo movement for cycle %d", i + 1)

        servo_controller._activate_servo(servo_array)  # MUST block until finished

        logger.info("Servo finished for cycle %d", i + 1)

        # --- SKIP MOTOR ON LAST LOOP ---
        if i == 11:
            logger.info("Last cycle reached. Motor will NOT be activated.")
            break

        # --- MOTOR + SENSOR RUN AFTER SERVO ---
        stop_event = threading.Event()

        motor_thread = threading.Thread(
            target=sensor_motor_control,
            args=(ir_sensor, motor_system, trigger_color, stop_event, button),
            daemon=True
        )

        logger.info("Starting motor thread for cycle %d", i + 1)
        motor_thread.start()

        # Wait until motor thread ends
        motor_thread.join()

        # Switch trigger color after motor stops
        trigger_color = "black" if trigger_color == "white" else "white"
        logger.info("Trigger color switched to %s", trigger_color)

        time.sleep(0.2)

    # 4. Set position backward
    motor_system.set_direction("BACKWARD")
    logger.info("Position set to BACKWARD")

    # Cleanup hardware for this run
    motor_system.stop()
    servo_controller.stop_all()
    ir_sensor.cleanup()
    motor_system.cleanup()

    logger.info("Run completed safely.")


# ---------------- MAIN PROGRAM ---------------- #
def main():
    logger.info("System booting...")

    button = buttons.ButtonController()

    try:
        while True:
            # Wait for START button
            button.wait_for_start()
            button.set_running(True)

            logger.info("START button accepted. Main loop starting...")

            try:
                run_main_loop(button)

            except Exception as e:
                logger.exception("Unexpected error during run: %s", str(e))

            finally:
                # Reset system after run or force stop
                button.set_running(False)
                button.reset()
                logger.info("System reset. Waiting for next START.")

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.warning("Program terminated by user (Ctrl+C).")

    finally:
        button.cleanup()
        GPIO.cleanup()
        logger.info("GPIO cleaned up. Program exited safely.")


if __name__ == "__main__":
    main()
