import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import os
from pathlib import Path
import logging

'''
EXAMPLE USAGE:
# In main.py
grid_data = self.model.run_inference()

# grid_data looks like this:
# {
#    1: ["G", "NG", "G", "Empty", ...], 
#    2: ["NG", "G", "G", "G", ...],
#    ...
#    12: [...]
# }
'''

# Initialize Logger to track system events and errors
logger = logging.getLogger(__name__)

class VisionSystem:
    """
    The VisionSystem class manages the computer vision pipeline for the Cocoon Sorting Machine.

    Responsibilities:
    1.  **Hardware Control**: manages the Raspberry Pi Camera.
    2.  **Object Detection**: Uses YOLOv8 to find cocoons and classify them (Good, NG, Empty).
    3.  **Grid Mapping**: Uses K-Means clustering to mathematically map scattered detections 
        into a structured 12x12 logical grid, regardless of slight camera misalignments.

    Attributes:
        camera_index (int): The ID of the camera (0 is usually the default USB/PiCam).
        model_path (str): Full file path to the .pt YOLO weights file.
        EXPECTED_ROWS (int): The physical number of rows in the montage (12).
        EXPECTED_COLS (int): The physical number of columns in the montage (12).
        CLASS_MAP (dict): Maps YOLO integer class IDs to human-readable labels ("NG", "G").
    """

    def __init__(self, model_name="best.pt", model_dir="models"):
        """
        Initialize the VisionSystem, load the AI model, and configure grid settings.

        Args:
            model_name (str): The filename of the trained YOLO model (default: "best.pt").
            model_dir (str): The folder containing the model file (default: "models").
        
        Raises:
            FileNotFoundError: If the model file does not exist at the specified path.
            Exception: If YOLO fails to load (e.g., corrupted file or missing library).
        """
        
        # Construct the absolute path to the model to avoid "file not found" errors on boot
        # using os.getcwd() ensures we find the file relative to where the script is run
        # base_dir = os.getcwd() 
        # self.model_path = os.path.join(base_dir, model_dir, model_name)

        # Get the folder of the current file (inference.py)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build the full path to the model
        self.model_path = os.path.join(base_dir, model_dir, model_name)
        

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        
        
        # --- 2. Configuration ---
        self.EXPECTED_ROWS = 12
        self.EXPECTED_COLS = 12
        
        # Mapping Class IDs (from training) to Labels.
        # CRITICAL: These IDs must match your data.yaml from Roboflow/YOLO training.
        self.CLASS_MAP = {
            0: "Empty",     # EMPTY
            1: "G",      # Good
            2: "NG"   # NG
        }

        # --- 3. Model Loading ---
        logger.info(f"Loading YOLO model from {self.model_path}...")
        try:
            # Load the model into memory. This can take 2-5 seconds on a Raspberry Pi.
            self.model = YOLO(self.model_path)
            logger.info("YOLO Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load YOLO model: {e}")
            self.model = None
            raise e

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
        """
        The Master Function: Orchestrates the entire Vision Pipeline.
        
        Steps:
        1. Capture Image.
        2. Predict using YOLO (Get list of random bounding boxes).
        3. Extract the Center X and Center Y of every box.
        4. Use K-Means to figure out which X belongs to which Column, and which Y to which Row.
        5. Fill a clean Dictionary with the results.

        Returns:
            dict: A dictionary where keys are Row Numbers (1-12) and values are lists of 12 strings.
                  Example: { 
                      1: ["G", "NG", "G", ...], 
                      2: ["Empty", "G", ...], 
                      ... 
                  }
        """
        # --- Step 1: Capture ---
        frame = self.capture_image()
        if frame is None:
            # If camera fails, return empty grid to prevent system crash
            return self._generate_empty_grid()

        # --- Step 2: YOLO Inference ---
        logger.info("Running YOLO Inference on captured frame...")
        # conf=0.25: Only accept detections with >25% confidence
        # verbose=False: Stops YOLO from spamming the console
        results = self.model(frame, conf=0.25, verbose=False)
        
        raw_detections = [] # To store temporary data: {cx, cy, label}
        all_x = []          # List of all X coordinates for clustering
        all_y = []          # List of all Y coordinates for clustering

        # --- Step 3: Parse Detections ---
        # results[0] contains the data for the first (and only) image
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Get coordinates (top-left, bottom-right)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate Centroid (Center of the cocoon)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Get Class ID and map it to string (0 -> "NG")
                cls_id = int(box.cls[0].item())
                label = self.CLASS_MAP.get(cls_id, "Unknown")
                
                raw_detections.append({
                    "cx": cx, 
                    "cy": cy, 
                    "label": label
                })
                all_x.append(cx)
                all_y.append(cy)
        else:
            logger.warning("YOLO detected 0 objects. Is the grid empty?")
            return self._generate_empty_grid()

        # --- Step 4: Spatial Clustering (The "Grid Logic") ---
        # We need to map the raw pixels to Rows (1-12) and Cols (1-12).
        
        # Check if we have enough data to form the grid. 
        # If we have fewer than 12 items, K-Means for 12 clusters will crash.
        if len(all_x) < 12 or len(all_y) < 12:
            logger.error("Not enough detections to establish grid pattern (Found <12 items).")
            return self._generate_empty_grid()

        try:
            # Get the mapping models
            kmeans_rows, row_ranks = self._get_cluster_map(all_y, self.EXPECTED_ROWS)
            kmeans_cols, col_ranks = self._get_cluster_map(all_x, self.EXPECTED_COLS)
        except Exception as e:
            logger.error(f"Clustering algorithm failed: {e}")
            return self._generate_empty_grid()

        # --- Step 5: Fill the Grid ---
        # Initialize a "Clean" grid full of "Empty"
        grid_output = {r: ["Empty"] * self.EXPECTED_COLS for r in range(1, self.EXPECTED_ROWS + 1)}

        if kmeans_rows and kmeans_cols:
            for det in raw_detections:
                # Ask KMeans: "Which cluster does this Y coordinate belong to?"
                y_cluster = kmeans_rows.predict([[det['cy']]])[0]
                # Convert Cluster ID -> Row Number (1-12)
                row_num = row_ranks[y_cluster]
                
                # Ask KMeans: "Which cluster does this X coordinate belong to?"
                x_cluster = kmeans_cols.predict([[det['cx']]])[0]
                # Convert Cluster ID -> Col Number (1-12)
                col_num = col_ranks[x_cluster]
                
                # Place label in the grid
                # Note: col_num is 1-based, list index is 0-based, so we use [col_num - 1]
                if 1 <= row_num <= 12 and 1 <= col_num <= 12:
                    grid_output[row_num][col_num - 1] = det['label']
        
        logger.info("Grid mapping and classification complete.")
        return grid_output


    def run_inference_from_folder(self, input_folder, output_folder):
        """
        Runs the Vision Pipeline on all images inside a folder.

        Args:
            input_folder (str): Path to folder containing images.
            output_folder (str): Path where annotated images will be saved.

        Returns:
            dict: {
                "image_name.jpg": {
                    1: [...],
                    2: [...],
                    ...
                },
                ...
            }
        """

        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(base_dir, '..')
        
        # Build the full path to the model
        output_folder_path = os.path.join(root_dir,output_folder)
        input_folder_path = os.path.join(root_dir,input_folder)


        os.makedirs(output_folder_path, exist_ok=True)

        supported_ext = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = [f for f in os.listdir(input_folder_path) if f.lower().endswith(supported_ext)]

        all_results = {}

        for image_name in image_files:
            image_path = os.path.join(input_folder_path, image_name)
            logger.info(f"Processing image: {image_name}")

            frame = cv2.imread(image_path)
            if frame is None:
                logger.warning(f"Failed to load image: {image_name}")
                continue

            # --- YOLO Inference ---
            results = self.model(frame, conf=0.25, verbose=False)

            raw_detections = []
            all_x = []
            all_y = []

            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    cls_id = int(box.cls[0].item())
                    label = self.CLASS_MAP.get(cls_id, "Unknown")

                    raw_detections.append({
                        "cx": cx,
                        "cy": cy,
                        "label": label
                    })

                    all_x.append(cx)
                    all_y.append(cy)

                # ✅ Save Annotated Image
                annotated_frame = results[0].plot()
                output_path = os.path.join(output_folder_path, image_name)
                cv2.imwrite(output_path, annotated_frame)

            else:
                logger.warning(f"No detections in image: {image_name}")
                all_results[image_name] = self._generate_empty_grid()
                continue

            # --- Grid Validation ---
            if len(all_x) < 12 or len(all_y) < 12:
                logger.error(f"Not enough detections in {image_name}")
                all_results[image_name] = self._generate_empty_grid()
                continue

            try:
                kmeans_rows, row_ranks = self._get_cluster_map(all_y, self.EXPECTED_ROWS)
                kmeans_cols, col_ranks = self._get_cluster_map(all_x, self.EXPECTED_COLS)
            except Exception as e:
                logger.error(f"Clustering failed for {image_name}: {e}")
                all_results[image_name] = self._generate_empty_grid()
                continue

            # --- Fill Grid ---
            grid_output = {
                r: ["Empty"] * self.EXPECTED_COLS
                for r in range(1, self.EXPECTED_ROWS + 1)
            }

            for det in raw_detections:
                y_cluster = kmeans_rows.predict([[det['cy']]])[0]
                row_num = row_ranks[y_cluster]

                x_cluster = kmeans_cols.predict([[det['cx']]])[0]
                col_num = col_ranks[x_cluster]

                if 1 <= row_num <= 12 and 1 <= col_num <= 12:
                    grid_output[row_num][col_num - 1] = det['label']

            all_results[image_name] = grid_output
            logger.info(f"Finished processing {image_name}")

        logger.info("Folder inference complete.")
        return all_results


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