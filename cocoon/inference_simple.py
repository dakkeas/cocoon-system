
import cv2
import numpy as np
import os
import glob
import time

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

class CocoonDetection:
    
    def __init__(self, model_path="best.tflite", conf_threshold=0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Load Model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.img_height = self.input_shape[1]
        self.img_width = self.input_shape[2]
        
        # Define your classes here
        self.classes = {0: "Empty", 1: "G", 2: "NG"}

    def _preprocess(self, image):
        """Resizes and normalizes image for the model."""
        img = cv2.resize(image, (self.img_width, self.img_height))
        
        # Check if model expects float (float32) or quantized (uint8)
        if self.input_details[0]['dtype'] == np.float32:
            img = img.astype(np.float32) / 255.0
            
        return np.expand_dims(img, axis=0)

    def _postprocess(self, pred_output, original_shape):
        """
        Parses raw model output into a list of detections.
        """
        orig_h, orig_w = original_shape[:2]
        output = np.squeeze(pred_output).T # Shape becomes [8400, 7] typically
        
        detections = []
        
        for row in output:
            # YOLO output: [x, y, w, h, score_class0, score_class1, ...]
            scores = row[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > self.conf_threshold:
                # Extract Box (x, y, w, h)
                x, y, w, h = row[0], row[1], row[2], row[3]
                
                # --- AUTO-SCALING LOGIC ---
                # If coordinates are normalized (0-1), scale by image size.
                # If coordinates are pixels (already scaled to model size), re-scale to original image.
                
                if x < 1.5 and w < 1.5:  # Likely normalized 0-1
                    x1 = int((x - w/2) * orig_w)
                    y1 = int((y - h/2) * orig_h)
                    x2 = int((x + w/2) * orig_w)
                    y2 = int((y + h/2) * orig_h)
                else: # Likely pixels relative to 640x640 (or model size)
                    x1 = int((x - w/2) / self.img_width * orig_w)
                    y1 = int((y - h/2) / self.img_height * orig_h)
                    x2 = int((x + w/2) / self.img_width * orig_w)
                    y2 = int((y + h/2) / self.img_height * orig_h)

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "label": self.classes.get(class_id, "Unknown"),
                    "conf": float(confidence)
                })
        
        return detections

    def _draw_annotations(self, image, detections):
        """Draws boxes and labels on the image."""
        annotated_img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['label']} {det['conf']:.2f}"
            
            # Color: Green for G, Red for NG/Empty
            color = (0, 255, 0) if det['label'] == 'G' else (0, 0, 255)
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Text background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated_img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return annotated_img

    def predict(self, image):
        """Runs inference on a single image array."""
        input_data = self._preprocess(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        detections = self._postprocess(output_data, image.shape)
        annotated_image = self._draw_annotations(image, detections)
        
        return detections, annotated_image

    # ==========================================
    # 1. LIVE WEBCAM
    # ==========================================
    def run_live_cam(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        print("Starting Live Inference... Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Run inference
            detections, annotated_frame = self.predict(frame)
            
            # Show result
            cv2.imshow("Live Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    # ==========================================
    # 2. FRAME CAPTURE (One Shot)
    # ==========================================
    def capture_frame(self, camera_index=0, save_path="captured_result.jpg"):
        cap = cv2.VideoCapture(camera_index)
        
        # Warmup
        print("Warming up camera...")
        for _ in range(10): cap.read()
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to capture image.")
            return None

        print("Capturing and processing...")
        detections, annotated_frame = self.predict(frame)
        
        cv2.imwrite(save_path, annotated_frame)
        print(f"Result saved to {save_path}")
        return detections

    # ==========================================
    # 3. BATCH FROM FOLDER
    # ==========================================
    def process_folder(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        image_files = glob.glob(os.path.join(input_folder, "*.*"))
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_exts]
        
        print(f"Found {len(image_files)} images in {input_folder}")
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            
            if frame is None: continue
            
            detections, annotated_frame = self.predict(frame)
            
            save_path = os.path.join(output_folder, f"annotated_{filename}")
            cv2.imwrite(save_path, annotated_frame)
            print(f"Processed {filename} -> {len(detections)} detections.")