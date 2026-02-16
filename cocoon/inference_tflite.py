import cv2
import numpy as np
import logging
import os
from sklearn.cluster import KMeans
try:
    # Use the lightweight runtime if available
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback for PC testing
    import tensorflow.lite as tflite

logger = logging.getLogger(__name__)

class VisionSystem:
    def __init__(self, model_name="cocoon_model_v2_int8.tflite", model_dir="cocoon/models/cocoon_model_v2_tflite"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, '..', model_dir, model_name)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # --- TFLite Setup ---
        # Load the TFLite model and allocate tensors.
        self.interpreter = tflite.Interpreter(model_path=self.model_path, num_threads=4)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Model expects: [Batch, Height, Width, Channels]
        self.input_shape = self.input_details[0]['shape']
        self.img_height = self.input_shape[1]
        self.img_width = self.input_shape[2]

        # --- Grid Config ---
        self.EXPECTED_ROWS = 12
        self.EXPECTED_COLS = 12
        self.CLASS_MAP = {0: "Empty", 1: "G", 2: "NG"}
        
        
    def _preprocess(self, frame):
        """Resize and normalize the image for the TFLite model."""
        img = cv2.resize(frame, (self.img_width, self.img_height))
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        return np.expand_dims(img, axis=0)     # Add batch dimension

    def capture_image(self, camera_index):
        """
        Captures a single high-resolution frame from the primary camera.

        Why High Res? 
        Cocoons are small. We set the camera to 1920x1080 (1080p) to ensure the 
        YOLO model has enough pixels to differentiate between 'Good' and 'NG' textures.

        Returns:
            numpy.ndarray: The captured image (OpenCV format), or None if capture fails.
        """
        logger.info("Opening camera interface...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error("Could not access the camera. Check ribbon cable or USB connection.")
            return None

        # Force high resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # --- Camera Warmup ---
        # The first few frames from a cold camera are often dark or blurry 
        # as the auto-exposure and auto-white-balance algorithms settle.
        # We grab 10 "dummy" frames to let the sensor stabilize.
        for _ in range(10):
            cap.read()
            
        ret, frame = cap.read()
        cap.release() # Immediately release camera resource so other apps (or cleanup) can use it.
        
        if not ret:
            logger.error("Camera opened but failed to return a frame.")
            return None
            
        logger.info("Image captured successfully.")
        return frame

    def _get_cluster_map(self, coordinates, n_clusters):
        """
        A Helper method that uses K-Means Clustering to map pixel coordinates to Grid Ranks.

        How this works:
        Imagine you have a list of Y-coordinates for all detections: [102, 105, 101, 300, 305, 302...].
        Even though 102 != 105, they belong to the same "Row". 
        
        1. K-Means groups these numbers into 'n_clusters' (e.g., 12 groups).
        2. We calculate the center of each group.
        3. We sort the centers (Smallest Y = Row 1, Largest Y = Row 12).
        4. We create a map: "If a point belongs to Cluster A, it is Row 1".

        Args:
            coordinates (list): A list of X or Y integers (e.g., all center_x values).
            n_clusters (int): The number of expected rows or columns (usually 12).

        Returns:
            tuple: (kmeans_model, rank_map_dict)
                   - kmeans_model: The trained SKLearn model to predict future points.
                   - rank_map_dict: A dictionary mapping {cluster_id: logical_rank (1-12)}.
        """
        # Safety Check: We can't find 12 rows if we only detected 5 items.
        if len(coordinates) < n_clusters:
            logger.warning(f"Not enough data points ({len(coordinates)}) to cluster into {n_clusters} groups.")
            return None, None
            
        # Reshape data for Scikit-Learn (needs 2D array: [[x1], [x2], ...])
        X = np.array(coordinates).reshape(-1, 1)
        
        # Run K-Means
        # n_init=10 means it will run the algorithm 10 times and pick the best center positions.
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(X)
        
        # Get the "centers" of the clusters (e.g., the average Y position of Row 1, Row 2, etc.)
        centers = kmeans.cluster_centers_.flatten()
        
        # Sort the centers to determine rank.
        # Example: The cluster with center Y=100 is Row 1. The cluster with center Y=900 is Row 12.
        sorted_indices = np.argsort(centers)
        
        # Create the mapping dictionary
        rank_map = {}
        for rank, cluster_id in enumerate(sorted_indices):
            # rank is 0-11, so we add 1 to match our physical 1-12 layout
            rank_map[cluster_id] = rank + 1
            
        return kmeans, rank_map

    def run_inference(self):
        frame = self.capture_image()
        if frame is None:
            return self._generate_empty_grid()

        # 1. Prepare Input
        input_data = self._preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # 2. Run Model
        self.interpreter.invoke()

        # 3. Get Raw Output
        # YOLOv8 TFLite output is usually [1, 7, 8400] (box_xywh + 3 classes)
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        output = np.squeeze(output).T # Shape: [8400, 7]

        raw_detections = []
        all_x, all_y = [], []

        # 4. Filter by Confidence & NMS
        # For YOLOv8/v11, first 4 cols are cx, cy, w, h. Rest are class scores.
        for pred in output:
            box = pred[:4]
            scores = pred[4:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if conf > 0.25:
                # Scale coordinates back to original frame size
                orig_h, orig_w = frame.shape[:2]
                cx = (box[0] / self.img_width) * orig_w
                cy = (box[1] / self.img_height) * orig_h
                
                label = self.CLASS_MAP.get(class_id, "Unknown")
                raw_detections.append({"cx": cx, "cy": cy, "label": label})
                all_x.append(cx)
                all_y.append(cy)

        # --- Step 5: Spatial Clustering (Same as your original code) ---
        if len(all_x) < 12 or len(all_y) < 12:
            return self._generate_empty_grid()

        try:
            kmeans_rows, row_ranks = self._get_cluster_map(all_y, self.EXPECTED_ROWS)
            kmeans_cols, col_ranks = self._get_cluster_map(all_x, self.EXPECTED_COLS)
            
            grid_output = {r: ["Empty"] * self.EXPECTED_COLS for r in range(1, 13)}

            for det in raw_detections:
                row_num = row_ranks[kmeans_rows.predict([[det['cy']]])[0]]
                col_num = col_ranks[kmeans_cols.predict([[det['cx']]])[0]]
                grid_output[row_num][col_num - 1] = det['label']
            
            return grid_output
        except Exception as e:
            logger.error(f"Grid mapping failed: {e}")
            return self._generate_empty_grid()
        
        
        
    def run_inference_from_folder(self, input_folder, output_folder):
        """
        Runs inference on all images in a folder, saves annotated images, 
        and returns a dictionary of results.
        
        Args:
            input_folder (str): Path to folder containing source images.
            output_folder (str): Path to folder where annotated images will be saved.
            
        Returns:
            dict: { "filename.jpg": grid_output_dict, ... }
        """
        import glob
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"Created output directory: {output_folder}")

        # Get all images (jpg, png, jpeg)
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        
        full_results = {}

        logger.info(f"Found {len(image_paths)} images in {input_folder}")

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            frame = cv2.imread(img_path)

            if frame is None:
                logger.warning(f"Could not read image: {filename}")
                continue

            # --- Core Inference Logic (Reused from run_inference) ---
            input_data = self._preprocess(frame)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            output = np.squeeze(output).T 

            raw_detections = []
            all_x, all_y = [], []
            orig_h, orig_w = frame.shape[:2]

            # Filter Confidence
            for pred in output:
                box = pred[:4] # cx, cy, w, h (normalized 0-1 if model output is normalized, or raw pixels)
                scores = pred[4:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                if conf > 0.25:
                    # Note: TFLite YOLO export usually outputs absolute pixels based on 640x640 input
                    # We need to scale them to the ORIGINAL image size
                    # Box coords come out relative to model input size (e.g. 640)
                    
                    # Scale factor (Original / Model Input)
                    scale_x = orig_w / self.img_width
                    scale_y = orig_h / self.img_height

                    # If the model output is normalized (0-1), remove the divisor. 
                    # Assuming standard YOLOv8 export where output is pixels relative to input_shape:
                    cx = (box[0] / self.img_width) * orig_w
                    cy = (box[1] / self.img_height) * orig_h
                    w  = (box[2] / self.img_width) * orig_w
                    h  = (box[3] / self.img_height) * orig_h

                    label = self.CLASS_MAP.get(class_id, "Unknown")
                    
                    raw_detections.append({
                        "cx": cx, "cy": cy, "w": w, "h": h, 
                        "label": label, "conf": conf
                    })
                    all_x.append(cx)
                    all_y.append(cy)

            # --- Clustering Logic ---
            grid_output = self._generate_empty_grid()
            
            # Only cluster if we have enough points, otherwise return empty grid for this file
            if len(all_x) >= 12 and len(all_y) >= 12:
                try:
                    kmeans_rows, row_ranks = self._get_cluster_map(all_y, self.EXPECTED_ROWS)
                    kmeans_cols, col_ranks = self._get_cluster_map(all_x, self.EXPECTED_COLS)
                    
                    # Assign grid positions to detections
                    for i, det in enumerate(raw_detections):
                        row_num = row_ranks[kmeans_rows.predict([[det['cy']]])[0]]
                        col_num = col_ranks[kmeans_cols.predict([[det['cx']]])[0]]
                        
                        # Update the result grid
                        grid_output[row_num][col_num - 1] = det['label']
                        
                        # Store grid pos in detection for annotation
                        raw_detections[i]['grid_pos'] = (row_num, col_num)
                except Exception as e:
                    logger.error(f"Clustering failed for {filename}: {e}")
            else:
                logger.warning(f"Skipping clustering for {filename}: Not enough detections ({len(all_x)})")

            # --- Annotation & Saving ---
            annotated_frame = self._draw_annotations(frame, raw_detections)
            save_path = os.path.join(output_folder, f"annotated_{filename}")
            cv2.imwrite(save_path, annotated_frame)
            
            full_results[filename] = grid_output
            print(f"Processed {filename} -> Saved to {save_path}")

        return full_results

    def _draw_annotations(self, frame, detections):
        """
        Helper to draw bounding boxes and grid coordinates on an image.
        """
        img = frame.copy()
        
        for det in detections:
            cx, cy, w, h = det['cx'], det['cy'], det['w'], det['h']
            label = det['label']
            conf = det['conf']
            
            # Convert center-x/y to top-left x/y for OpenCV
            x1 = int(cx - (w / 2))
            y1 = int(cy - (h / 2))
            x2 = int(cx + (w / 2))
            y2 = int(cy + (h / 2))
            
            # Color based on class
            color = (0, 255, 0) if label == "G" else (0, 0, 255) # Green for G, Red for NG
            
            # Draw Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label (Class + Grid Pos if available)
            grid_str = ""
            if 'grid_pos' in det:
                grid_str = f"R{det['grid_pos'][0]}-C{det['grid_pos'][1]}"
                
            text = f"{label} {conf:.2f} {grid_str}"
            
            # Text Background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return img              

    def _generate_empty_grid(self):
        """
        Fallback Method.
        Creates a 'safe' 12x12 grid filled with "Empty".
        Used when the camera fails, model fails, or no objects are seen, 
        ensuring the main program loop doesn't crash due to missing data.
        """
        return {r: ["Empty"] * self.EXPECTED_COLS for r in range(1, self.EXPECTED_ROWS + 1)}

    def check_camera(self):
        """
        Perform a system health check. Used by the main orchestrator on boot.
        
        Checks:
        1. Is the Camera accessible?
        2. Was the Model loaded into memory?

        Returns:
            bool: True if systems are go, False if critical failure.
        """
        # Check Camera
        cap = cv2.VideoCapture(self.camera_index)
        cam_status = cap.isOpened()
        cap.release()
        
        # Check Model

        if not cam_status:
            logger.error("Health Check Failed: Camera not found.")

        return cam_status 

    def check_model(self):
        """
        Perform a system health check. Used by the main orchestrator on boot.
        
        Checks:
        1. Is the Camera accessible?
        2. Was the Model loaded into memory?

        Returns:
            bool: True if systems are go, False if critical failure.
        """
        
        # Check Model
        model_status = self.model is not None

        if not model_status:
            logger.error("Health Check Failed: Model not loaded.")

        return model_status