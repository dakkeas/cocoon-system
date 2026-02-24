import threading
import time
import logging
import RPi.GPIO as GPIO
import requests 
import json

# from .cocoon.hardware import buttons
from cocoon.hardware import motor
from cocoon.hardware import sensor
from cocoon.hardware import motor_sensor
from cocoon.hardware import servo
from cocoon.hardware import buttons
# from cocoon.hardware import camera
from cocoon import flask
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
# ir_motor = motor_sensor.IRMotorController(motor_system, ir_sensor)
# camera = camera.CameraManager()  # Initialize camera with index 0
servo_controller = servo.ServoController()
button_controller = buttons.ButtonController()
client = flask.FlaskAPIClient("http://localhost:5000")
motor_speed = 40


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

# ... (previous imports and setup)

# # ---------------- MAIN LOOP LOGIC ---------------- #
# def run_main_loop(button):
#     logger.info("All systems operational.")
#     client.send_log("All systems operational. Starting main loop.") # Sync log

#     motor_system.set_direction("FORWARD")
    
#     # 1. Run Inference
#     client.send_log("📸 Capturing image and running AI inference...")
#     test_result_cam = vision_system.run_inference() 
#     # Use the actual inference results
#     results = test_result_cam if test_result_cam else vision_system._generate_empty_grid()
    
#     client.send_log("Inference completed. Mapping grid...")
#     # Optionally send the full JSON here once, or rely on row-by-row
#     client.send_json(results)

#     for loop_idx in range(1, 13): 
#         logger.info(f"Starting cycle {loop_idx} / 12")
        
#         # --- NEW: Send row-specific data to Flask per cycle ---
#         row_data = results.get(loop_idx) or results.get(str(loop_idx))
#         client.send_json({"row_update": {str(loop_idx): row_data}})
#         client.send_log(f"Starting cycle {loop_idx} / 12")

#         if button.should_stop():
#             logger.warning("Force stop detected.")
#             client.send_log("STOP button pressed. Terminating...")
#             break

#         target_color = 0 if loop_idx % 2 == 1 else 1
#         ir_motor = motor_sensor.IRMotorController(motor_system, ir_sensor, target_color, speed=40)
#         ir_motor.start()
#         ir_motor.wait_until_finished()
#         motor_system.stop()

#         servo_array = row_data
#         servo_controller._activate_servos(servo_array)
        
#         time.sleep(1) # Reduced sleep for better responsiveness

#     motor_system.set_direction("BACKWARD")
#     motor_system.start(motor_speed)
#     time.sleep(0.5)
#     motor_system.stop()
#     client.send_log("🏁 Run completed safely. Returning to IDLE.")

# # ---------------- MAIN PROGRAM ---------------- #

# def main():
#     # Sync boot logs
#     client.send_log("⚙️ System booting...")
#     logger.info("System booting...")

#     try:
#         while True:
#             client.send_log("⌛ Waiting for START signal...")
#             logger.info("Waiting for START button...")
            
#             button_controller.wait_for_start()
            
#             client.send_log("▶️ START signal detected!")
#             button_controller.set_running(True)
#             run_main_loop(button_controller)

#             button_controller.set_running(False)
#             button_controller.reset()
#             time.sleep(0.5)

#     except KeyboardInterrupt:
#         client.send_log(" Program terminated manually.")
# # ... rest of file
 
# --- NEW FUNCTION: LISTENS FOR WEB BUTTON PRESSES ---
def poll_web_commands(client, btn_controller):
    logger.info("Web Listener active.")
    while True:
        try:
            cmd = client.get_command() # checks app.py for commands
            if cmd == 'start':
                logger.info("🌐 Web START received.")
                btn_controller.start_event.set() # Triggers the physical start logic
            elif cmd == 'stop':
                logger.info("🌐 Web STOP received.")
                btn_controller.stop_event.set()  # Triggers the physical stop logic
        except:
            pass
        time.sleep(0.5)
        
# ---------------- MAIN LOOP LOGIC ---------------- #
def run_main_loop(button):
    logger.info("All systems operational.")
    client.send_log("System started. Running main loop.")

    motor_system.set_direction("FORWARD")
    
    # test_result_cam = vision_system.run_inference()
    results = vision_system._generate_empty_grid()# get 12x4 array of servo activations
    print('sending results to Flask')
    client.send_log("Sending inference")
    client.send_json(results)

    logger.info("Inference completed.")

    for loop_idx in range(1,13): # 12 cycles total (1-12)
        logger.info(f"Starting cycle {loop_idx} / 12")
        client.send_log(f"Starting cycle {loop_idx} / 12")

        # STOP button check
        if button.should_stop():
            logger.warning("Force stop detected. Exiting run loop.")
            break

        # -------------------- IR SENSOR TARGET COLOR -------------------- #
        # For each of the 12 cycles, the motor should run while the IR sensor
        # detects a specific surface color. The logic alternates every cycle:
        #   - Even-numbered cycles (0, 2, 4, ...) → motor runs while BLACK (1)
        #   - Odd-numbered cycles (1, 3, 5, ...)  → motor runs while WHITE (0)
        # The IRMotorController thread will monitor the sensor and stop the motor
        # automatically when the color no longer matches.
        target_color = 0 if loop_idx % 2 == 1 else 1

        # Start IR motor controller thread
        ir_motor = motor_sensor.IRMotorController(motor_system, ir_sensor, target_color, motor_speed)
        ir_motor.start()

        # Wait until IR motor controller finishes this cycle
        ir_motor.wait_until_finished()

        motor_system.stop()  # ensure motor is stopped

        # --- Servo operations (optional) ---
        servo_array = results[loop_idx]
        print("Servo activations for this cycle:", servo_array)
        print(results[loop_idx])
        logger.info(f"Starting servo for cycle {loop_idx}")
        time.sleep(3)
        servo_controller._activate_servos(servo_array)
        logger.info(f"Servo finished for cycle {loop_idx}")

        time.sleep(3)  # small delay before next cycle

    # Reverse motor direction at end
    motor_system.set_direction("BACKWARD")
    motor_system.start(motor_speed)
    time.sleep(0.5)  # run back motor to return to start
    motor_system.stop()

    logger.info("Run completed safely.")
 
# ---------------- MAIN PROGRAM ---------------- #

def main():
    logger.info("System booting...")


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