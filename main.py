import threading
import time
import logging
import RPi.GPIO as GPIO
import requests 
import json

# from .cocoon.hardware import buttons
from cocoon.hardware import motor
from cocoon.hardware import sensor
# from cocoon.hardware import servo
from cocoon.hardware import buttons
# from cocoon.hardware import camera
# from cocoon import flask
from cocoon import inference

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

# ---------------- MOTOR THREAD (TIMED) ---------------- #


vision_system = inference.VisionSystem()
motor_system = motor.MotorSystem()
ir_sensor = sensor.IR_Sensor()
# camera = camera.CameraManager()  # Initialize camera with index 0
# servo_controller = servo.ServoController()
button_controller = buttons.ButtonController()
# client = flask.FlaskAPIClient("http://localhost:5000")


# ---------------- CAMERA FEED LOGIC ---------------- #
# def send_live_frame(camera, client):
#     frame = camera.get_frame()
#     if frame is None:
#         return False

#     # overwrite same file each time
#     live_frame_path = "../output/live_frame.jpg"
#     cv2.imwrite(live_frame_path, frame)

#     # send to Flask
#     client.send_live_frame(live_frame_path)
#     return True


# def live_feed_loop(camera, client):
#     while True:
#         send_live_frame(camera, client)
#         time.sleep(0.1)  # 10 FPS
    

# ---------------- MAIN LOOP LOGIC ---------------- #
def run_main_loop(button):
    logger.info("All systems operational.")

    # client.send_log("All systems operational")
    time.sleep(0.5)  # Short delay to ensure log is sent before inference starts
    # client.send_log("Main loop started. Running inference and motor control.")
    # Upload log

    motor_system.set_direction("FORWARD")

    # results = vision_system.run_inference()
    results = vision_system._generate_empty_grid()
    # client.send_json(results)
    # client.send_log("Inference results sent to server.")


    print("Inference results:")
    print(results)
    logger.info("Inference completed.")
    

    normal_motor_run_duration = 0.075
    halfway_motor_run_duration = 0.0752
    run_back_motor_run_duration = 0.625
    motor_speed = 10
    # client.send_log("Running motors & servos based on inference results.")

    for i in range(12):
        if i == 0 or i == 11:
            logger.info("Initial cycle. Running motor for first time to set position to first row.")
            motor_system.start(70)
            time.sleep(0.09)  # Run motor for 100ms to ensure it
            motor_system.stop()

        logger.info("Cycle %d / 12 started", i + 1)

        # STOP button check
        if button.should_stop():
            logger.warning("Force stop detected. Exiting run loop.")
            break

        servo_array = results[i+1]

        # --- SERVO ---
        logger.info("Starting servo for cycle %d", i + 1)
        # servo_controller._activate_servo(servo_array)
        logger.info("Servo finished for cyce %d", i + 1)

        if i == 0 or i == 11:
            time.sleep(2) 
        else:
            time.sleep(1)

        # Skip motor on last cycle
        if i == 11:
            logger.info("Last cycle reached. Motor will not run.")
            break

        # --- MOTOR ---
        logger.info("Starting motor for cycle %d", i + 1)
        motor_system.start(70)
        if i == 5:  # halfway point
            logger.info("Halfway point reached. Running motor for extended duration.")
            time.sleep(halfway_motor_run_duration)
        else:
            time.sleep(normal_motor_run_duration)

        motor_system.stop()
        logger.info("Motor finished for cycle %d", i + 1)

        time.sleep(1)

    # Reverse direction after loop
    # motor_system.direction = 'BACKWARD' if motor_system.direction == 'FORWARD' else 'FORWARD'
    # motor_system.set_direction(motor_system.direction)

    # logger.info(f"Direction set to {motor_system.direction}")
    time.sleep(1)
    motor_system.set_direction("BACKWARD")
    motor_system.start(70)
    time.sleep(run_back_motor_run_duration)
    motor_system.stop()
    # servo_controller.stop_all()

    logger.info("Run completed safely.")
# ---------------- MAIN PROGRAM ---------------- #

def main():
    logger.info("System booting...")

    # live_feed_thread = threading.Thread(target=live_feed_loop, args=(camera, client), daemon=True)
    # live_feed_thread.start()

    try:
        while True:
            # Wait for START
            logger.info("Waiting for START button...")
            button_controller.wait_for_start()
            button_controller.set_running(True)

            logger.info("START pressed. Running main loop.")

            try:
                run_main_loop(button_controller)

            except Exception as e:
                logger.exception("Unexpected error: %s", str(e))

            finally:
                # Always reset after run or stop
                button_controller.set_running(False)
                button_controller.reset()
                logger.info("System reset. Waiting for next START.")

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.warning("Program terminated by user.")

    finally:
        button_controller.cleanup()
        GPIO.cleanup()
        logger.info("GPIO cleaned up. Program exited safely.")

if __name__ == "__main__":
    main()