from cocoon import VisionSystem


def main():
    
    print('running main loop')
    
    model = VisionSystem(
        model_name ='cocoon_model_v2.pt',
        model_dir ='models'
    )

    
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
    

if __name__ == "__main__": 
    main()












