from cocoon.hardware import motor
import time
import config


motor_system = motor.MotorSystem()

try:
    # Initialize the "Super Motor"
    # The class will detect that all PWMs are Pin 17 and only create one controller
    
    # direction = 'FORWARDource
    # direction = 'FORWARD'
    while True:

        print(f'-----------------------> RUNNING AT 10') 
        # time.sleep(2)
        motor_system.start(10) # dito na yung first actuation sa first row
        time.sleep(3)
        motor_system.stop()
        print(f'-----------------------> RUNNING AT 15') 
        motor_system.start(15) # dito na yung first actuation sa first row
        time.sleep(3)
        motor_system.stop()
        print(f'-----------------------> RUNNING AT 20') 
        motor_system.start(20) # dito na yung first actuation sa first row
        time.sleep(3)
        motor_system.stop()
        print(f'-----------------------> RUNNING AT 30') 
        motor_system.start(30) # dito na yung first actuation sa first row
        time.sleep(3)
        motor_system.stop()
        print(f'-----------------------> RUNNING AT 40') 
        motor_system.start(40) # dito na yung first actuation sa first row
        time.sleep(3)
        motor_system.stop()
        print(f'-----------------------> RUNNING AT 50') 
        motor_system.start(50) # dito na yung first actuation sa first row
        time.sleep(3)
        motor_system.stop()

        # time.sleep(8)
        


except KeyboardInterrupt:
    print("Stopped by user")

finally:
    motor_system.cleanup()
