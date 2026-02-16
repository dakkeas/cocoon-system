from cocoon import inference_simple
import os

# 1. Initialize
# Make sure your model file is in the same folder or provide full path
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir,'..','cocoon/models/cocoon_model_v2_tflite/cocoon_model_v2_float32.tflite')
test_images = os.path.join(base_dir, '..', 'test_dataset')
output = os.path.join(base_dir, '..', 'output')


vision = inference_simple.CocoonDetection(model_path=model_path)

# --- OPTION A: Run Live Webcam ---
# vision.run_live_cam(0)

# --- OPTION B: Capture Single Frame ---
# vision.capture_frame(camera_index=0, save_path="my_capture.jpg")

# --- OPTION C: Run on Folder ---
vision.process_folder(input_folder=test_images, output_folder=output)