from cocoon import VisionSystem
from cocoon.hardware import ServoController
import time


def main():
    
    print('running main loop')
    
    #model = VisionSystem(
    #    model_name ='cocoon_model_v2.pt',
    #    model_dir ='models'
    #)

    servo = ServoController()

    
    #if not model.check_model():
    #    print('system health check failed')
    #    return
    
    # print(model.check_camera())

    print('model is healthy....')

    print('running inference....')

    # grid_result = model.run_inference()

    # print('\n printing result ......')

    # print('\n IMAGE 1')
# 
    # print(grid_result[0])

    # print('\n IMAGE 2')

    # print(grid_result[0])
    grid_result = {1: ['Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty', 'G', 'G', 'NG', 'G', 'G', 'Empty'], 2: ['G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G'], 3: ['G', 'G', 'G', 'G', 'G', 'NG', 'G', 'G', 'G', 'G', 'G', 'Empty'], 4: ['G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'Empty'], 5: ['G', 'G', 'G', 'G', 'NG', 'G', 'G', 'G', 'G', 'G', 'G', 'Empty'], 6: ['G', 'G', 'G', 'G', 'G', 'G', 'NG', 'G', 'G', 'G', 'Empty', 'NG'], 7: ['G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'Empty'], 8: ['G', 'G', 'G', 'G', 'G', 'NG', 'G', 'NG', 'G', 'G', 'G', 'Empty'], 9: ['NG', 'G', 'G', 'NG', 'G', 'NG', 'G', 'G', 'NG', 'G', 'G', 'Empty'], 10: ['G', 'NG', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'Empty'], 11: ['G', 'G', 'G', 'G', 'G', 'NG', 'G', 'G', 'G', 'G', 'G', 'Empty'], 12: ['G', 'G', 'NG', 'G', 'G', 'G', 'G', 'Empty', 'Empty', 'Empty', 'Empty', 'Empty']}


    print(grid_result)

    for i in range(12):
        
        print(f'-------> running row {i+1}')
        print(f'-------> activating: {grid_result[i+1]}')
        servo_array = grid_result[i+1]
        time.sleep(5)
        servo._activate_servos(servo_array)

        print(f'-------> completed row {i+1} , moving to next')
    

if __name__ == "__main__": 
    main()












