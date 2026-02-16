from cocoon.hardware import motor
import time
import config



try:
    # Initialize the "Super Motor"
    # The class will detect that all PWMs are Pin 17 and only create one controller
    motor_system = motor.MotorSystem(config.MOTOR_CONFIG)
    
    print("Moving Forward...")
    motor_system.forward(80) 
    time.sleep(2)
    
    print("Moving Backward...")
    motor_system.backward(80)
    time.sleep(2)
    
    print("Stopping...")
    motor_system.stop()

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    motor_system.cleanup()
