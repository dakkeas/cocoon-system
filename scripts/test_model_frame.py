from cocoon import VisionSystem
from cocoon.hardware import ServoController
import time


def main():
    
    print('running main loop')
    
    model = VisionSystem(
        model_name ='cocoon_model_v2.pt',
        model_dir ='models'
    )

    servo = ServoController()

    
    if not model.check_model():
        print('system health check failed')
        return
    
    print(model.check_camera())

    print('model is healthy....')

    print('running inference....')

    grid_result = model.run_inference()

    # print('\n printing result ......')

    # print('\n IMAGE 1')
# 
    # print(grid_result[0])

    # print('\n IMAGE 2')

    # print(grid_result[0])


    print(grid_result)

    for i in range(12):
        
        print(f'-------> running row {i}')
        print(f'-------> {servo_array[i]}')
        servo_array = grid_result[i]

        servo._activate_servos(servo_array)

        time.sleep(1)

        print(f'-------> completed row {i}')
    

if __name__ == "__main__": 
    main()












