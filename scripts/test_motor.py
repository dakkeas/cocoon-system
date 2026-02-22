from cocoon.hardware import motor
import time
import config


motor_system = motor.MotorSystem()

try:
    # Initialize the "Super Motor"
    # The class will detect that all PWMs are Pin 17 and only create one controller
    
    # direction = 'FORWARDource
    direction = 'FORWARD'
    for x in range(5):
        motor_system.set_direction(direction)

        

        print(f'-----------------------> Moving to FIRST ROW POSITION') 
        time.sleep(2)
        motor_system.start(70) # dito na yung first actuation sa first row
        time.sleep(.1)
        motor_system.stop()

        # time.sleep(8)
        

        for y in range(1,12): # 12 instances (binawasa ko)
            
            print(f"----------------------> Activating SERVO (mock)") # need natin to ilipat
            time.sleep(2)
            print(f"----------------------> Moving to row {y + 1}...")
            motor_system.start(70) 
            time.sleep(0.083)
            motor_system.stop()

        print(f'-----------------------> Moving to FREE AREA') 
        time.sleep(1)
        motor_system.start(70) 
        time.sleep(0.13)
        motor_system.stop()

        time.sleep(2)

        direction = 'BACKWARD' if direction == 'FORWARD' else 'FORWARD'
        print(f'-------------> Direction set to {direction}')



except KeyboardInterrupt:
    print("Stopped by user")

finally:
    motor_system.cleanup()
