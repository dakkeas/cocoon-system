
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

    print('model is healthy....')

    grid_result = model.run_inference_from_folder('test_dataset','output')

    # print('\n printing result ......')

    # print('\n IMAGE 1')
# 
    # print(grid_result[0])

    # print('\n IMAGE 2')

    # print(grid_result[0])

    first_image = list(grid_result.keys())[3]
    print("First image:", first_image)

    # Get its dictionary
    first_image_dict = grid_result[first_image]

    # Print all rows inside that image
    for row_num, row_data in first_image_dict.items():
        print(f"Row {row_num}: {row_data}")


    
    

if __name__ == "__main__": 
    main()