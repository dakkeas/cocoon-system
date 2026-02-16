from ultralytics import YOLO
import os
# Load your model

base_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(base_dir, 'cocoon/models/cocoon_model_v2.pt'))

# Export to TFLite with INT8 quantization
model.export(format="tf")