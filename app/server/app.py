import sys
import os
import platform
import json
import time
import glob
import cv2  # Added for Live View
from datetime import datetime
from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS

# --- SMART PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
cocoon_dir = os.path.dirname(os.path.dirname(current_dir)) 
output_dir = os.path.join(cocoon_dir, "../output")
project_root = os.path.dirname(cocoon_dir)

sys.path.append(project_root)
sys.path.append(cocoon_dir)

OUTPUT_DIR = os.path.join(cocoon_dir, "output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

IS_WINDOWS = platform.system() == "Windows"

# --- IMPORTS ---
try:
    import config
    from cocoon.inference import VisionSystem
    
    # We will try to load hardware, but if it fails, we fall back to Mocks
    HARDWARE_AVAILABLE = False
    if not IS_WINDOWS:
        try:
            from cocoon.hardware.motor import MotorSystem
            from cocoon.hardware.servo import ServoController
            from cocoon.hardware.sensor import IR_Sensor
            HARDWARE_AVAILABLE = True
            print("✅ Hardware Libraries Found")
        except Exception as e:
            print(f"⚠️ Hardware libs found but failed to init: {e}")
    
    # Define Mocks if hardware is missing or on Windows
    if IS_WINDOWS or not HARDWARE_AVAILABLE:
        class MotorSystem:
            def __init__(self, *args): pass
            def forward(self, s): print(f"[MOCK] Motor FWD {s}")
            def stop(self): print("[MOCK] Motor STOP")
        class ServoController:
            def __init__(self, *args): pass
            def start(self, data): print(f"[MOCK] Servos Active: {data}")
        class IR_Sensor:
            def __init__(self): pass
            def read(self): return 0

except ImportError as e:
    print(f"\n❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# --- INITIALIZATION ---
vision = VisionSystem(model_name="cocoon_model_v2.pt", model_dir=os.path.join(cocoon_dir, "models"), camera_index=config.CAMERA_INDEX)

# Mock Fallbacks
class MockSystem:
    def __init__(self, *args, **kwargs): pass
    def forward(self, *args): print("[MOCK] Motor Moving")
    def stop(self): print("[MOCK] Motor Stopped")
    def start(self, *args): print("[MOCK] Servos Sequence Active")
    def read(self): return 0

# Safe Hardware Initialization
try:
    if HARDWARE_AVAILABLE and not IS_WINDOWS:
        servos = ServoController()
        motor = MotorSystem(config) 
        ir_sensor = IR_Sensor()
        print("🔌 Real Hardware Initialized")
    else:
        raise Exception("Hardware not requested or unavailable")
except Exception as e:
    print(f"🤖 Hardware skipped/failed ({e}). Using MOCK classes.")
    servos = MockSystem()
    motor = MockSystem()
    ir_sensor = MockSystem()

# --- GLOBAL STATE ---
system_state = {
    "sorting_active": False,
    "cocoon_grid": [0] * 144, 
    "stats": { "g_count": 0, "ng_count": 0, "empty_count": 0 },
    "last_scan_time": None,
    "logs": []  # NEW: Stores system logs
}

# --- HELPER FUNCTIONS ---
def add_log(message):
    """Adds a timestamped log to the global state."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry) # Print to console
    # Insert at the top (newest first)
    system_state["logs"].insert(0, entry)
    # Keep only last 50 logs
    if len(system_state["logs"]) > 50:
        system_state["logs"].pop()

def generate_frames():
    """Generator for the live camera feed."""
    camera = cv2.VideoCapture(config.CAMERA_INDEX) 
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize for faster transmission if needed
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()

# --- API ROUTES ---

@app.route('/api/data', methods=['GET'])
def get_data():
    stats = system_state["stats"]
    total = stats["g_count"] + stats["ng_count"] + stats["empty_count"]
    real_cocoons = stats["g_count"] + stats["ng_count"]
    rate = (stats["ng_count"] / real_cocoons * 100) if real_cocoons > 0 else 0

    return jsonify({
        "grid": system_state["cocoon_grid"],
        "g_count": stats["g_count"],
        "ng_count": stats["ng_count"],
        "empty_count": stats["empty_count"],
        "total": total,
        "defect_rate": round(rate, 1),
        "active": system_state["sorting_active"],
        "last_scan_time": system_state["last_scan_time"],
        "logs": system_state["logs"] # Send logs to frontend
    })

@app.route('/api/video_feed')
def video_feed():
    """Route for the live video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/latest_image')
def get_latest_image():
    """Serves the latest inferred result image."""
    try:
        list_of_files = glob.glob(os.path.join(OUTPUT_DIR, 'frame_inferred_*.jpg'))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            return send_file(latest_file, mimetype='image/jpeg')
        
        # Fallback image
        test_dummy = os.path.join(OUTPUT_DIR, "test.jpg")
        if os.path.exists(test_dummy):
            return send_file(test_dummy, mimetype='image/jpeg')
            
        return "No image found", 404
    except Exception as e:
        return str(e), 500

@app.route('/api/action', methods=['POST'])
def handle_action():
    data = request.get_json()
    action = data.get('action')
    
    if action == 'start':
        add_log("▶ START COMMAND RECEIVED")
        system_state["sorting_active"] = True
        
        try:
            add_log("📷 Vision System: Capturing & Inferring...")
            # 1. AI Classification
            grid_dict = vision.run_inference() 
            add_log("🧠 Inference Complete. Processing Grid Data...")
            
            # 2. Initialize temporary structures
            flat_grid = []
            coord_map = {}
            matrix_view = {}
            g, ng, empty = 0, 0, 0
            
            # 3. Process the 12x12 Grid
            for r in range(1, 13):
                row_items = []
                row_data = grid_dict.get(r, ["Empty"]*12)
                
                for c_idx, val in enumerate(row_data):
                    col = c_idx + 1
                    status_num = 1 if val == "G" else 2 if val == "NG" else 3
                    flat_grid.append(status_num)
                    coord_map[f"[{r},{col}]"] = val
                    
                    if val == "G": g += 1
                    elif val == "NG": ng += 1
                    else: empty += 1
                    
                    display_val = " G " if val == "G" else "NG " if val == "NG" else " E "
                    row_items.append(f"[ {display_val} ]")

                row_key = f"Row {r:02d}"
                matrix_view[row_key] = " ".join(row_items)

            # 4. Update Global State
            system_state["cocoon_grid"] = flat_grid
            system_state["stats"] = {"g_count": g, "ng_count": ng, "empty_count": empty}
            system_state["coordinate_map"] = coord_map
            system_state["matrix_view"] = matrix_view
            system_state["last_scan_time"] = time.time() 

            # 5. Save Master JSON
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = os.path.join(OUTPUT_DIR, f"scan_{ts}.json")
            with open(json_filename, 'w') as f:
                json.dump(system_state, f, indent=4)

            system_state["sorting_active"] = False
            add_log(f"✅ Scan Finished: {g} Good, {ng} Reject. Saved to JSON.")

        except Exception as e:
            add_log(f"❌ Scan Failed: {str(e)}")
            system_state["sorting_active"] = False

    elif action == 'reset':
        add_log("♻️ Batch Reset Initiated.")
        system_state["cocoon_grid"] = [0] * 144
        system_state["stats"] = { "g_count": 0, "ng_count": 0, "empty_count": 0 }
        system_state["coordinate_map"] = {}
        system_state["matrix_view"] = {}
        system_state["last_scan_time"] = None
        system_state["logs"] = [] # Clear logs
        add_log("System Ready. Logs Cleared.")

    return jsonify({"status": "success"})

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Threaded=True is important for the video feed to work alongside requests
    app.run(host='0.0.0.0', port=5000, threaded=True)