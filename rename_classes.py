from ultralytics import YOLO
import torch
# Load the 
model = YOLO(r"cocoon/models/cocoon_model_v2.pt")

# Check current class names
print("Old class names:", model.names)  # works even if model.model is Sequential

# Build a new names dict mapping indices to your preferred labels
new_names = {0: "ng", 1: "g", 2: "Empty"}

# Update the internal names dictionary safely
if hasattr(model.model, "names"):
    model.model.names = new_names
else:
    # fallback for Sequential or weird YOLOv8 export
    # overwrite the read-only property using __dict__ (hacky but works)
    model.__dict__['_model'].names = new_names

# Confirm
print("New class names:", model.names)
torch.save(model.model, r"cocoon/models/cocoon_model_v2_renamed.pt")