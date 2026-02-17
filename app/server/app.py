import sys
import os
import platform
import json
import time
import glob
from datetime import datetime
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# --- SMART PATH SETUP ---
# Current: cocoon/app/server/app.py
current_dir = os.path.dirname(os.path.abspath(__file__)) 
# Go up 2 levels to get 'cocoon' folder
cocoon_dir = os.path.dirname(os.path.dirname(current_dir)) 
# Go up 1 more to get project root (if needed)
project_root = os.path.dirname(cocoon_dir)

# Add paths so imports work
sys.path.append(project_root)
sys.path.append(cocoon_dir)

# Where inference.py saves images
OUTPUT_DIR = os.path.join(cocoon_dir, "output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

IS_WINDOWS = platform.system() == "Windows"

# --- IMPORTS ---
try:
    import config
    # from cocoon.utils.inference import VisionSystem # Updated Path based on structure
    
    if not IS_WINDOWS:
        from cocoon.hardware.motor import MotorSystem
        from cocoon.hardware.servo import ServoController
        from cocoon.hardware.sensor import IR_Sensor
        print("✅ Hardware Libraries Loaded")
    else:
        print("⚠️ Windows detected: Using Mock Hardware")
        class MotorSystem:
            def __init__(self, *args): pass
            def forward(self, s): print(f"[MOCK] Motor FWD {s}")
            def backward(self, s): print(f"[MOCK] Motor BWD {s}")
            def stop(self): print("[MOCK] Motor STOP")
        class ServoController:
            def __init__(self, *args): pass
            def start(self, data): print(f"[MOCK] Servos Active: {data}")
        class IR_Sensor:
            def __init__(self): pass
            def read(self): return 0 # Simulate detecting white line immediately

except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print(f"Debug: Looking in {cocoon_dir}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# --- GLOBAL STATE ---
system_state = {
    "sorting_active": False,
    "cocoon_grid": [0] * 144, 
    "stats": { "g_count": 0, "ng_count": 0, "empty_count": 0 },
    "last_scan_time": None # Helps frontend know when to refresh image
}

# --- INITIALIZATION ---
# Update model path to be absolute based on calculated root
model_abs_path = os.path.join(cocoon_dir, "models")
vision = VisionSystem(model_name=config.MODEL_NAME, model_dir=model_abs_path, camera_index=config.CAMERA_INDEX)

if not IS_WINDOWS:
    servos = ServoController()
    # config might store pins as simple vars, checking compatibility
    motor = MotorSystem(config) 
    ir_sensor = IR_Sensor()
else:
    servos = ServoController()
    motor = MotorSystem()
    ir_sensor = IR_Sensor()

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
        "last_scan_time": system_state["last_scan_time"]
    })

@app.route('/api/latest_image')
def get_latest_image():
    """Finds the most recent 'frame_inferred_*.jpg' and serves it."""
    try:
        # Look for inferred images
        list_of_files = glob.glob(os.path.join(OUTPUT_DIR, 'frame_inferred_*.jpg'))
        if not list_of_files:
            return "No image found", 404
        
        # Get the latest file
        latest_file = max(list_of_files, key=os.path.getctime)
        return send_file(latest_file, mimetype='image/jpeg')
    except Exception as e:
        return str(e), 500

@app.route('/api/action', methods=['POST'])
def handle_action():
    data = request.get_json()
    action = data.get('action')
    
    if action == 'start':
        print(f"\n▶ START COMMAND RECEIVED")
        system_state["sorting_active"] = True
        
        try:
            # 1. AI Classification
            grid_dict = vision.run_inference() 
            
            # 2. Update Stats
            flat_grid = []
            g, ng, empty = 0, 0, 0
            
            for r in range(1, 13):
                row_data = grid_dict.get(r, ["Empty"]*12)
                for val in row_data:
                    flat_grid.append(1 if val == "G" else 2 if val == "NG" else 3)
                
                g += row_data.count("G")
                ng += row_data.count("NG")
                empty += row_data.count("Empty")

            # Update State
            system_state["cocoon_grid"] = flat_grid
            system_state["stats"] = {"g_count": g, "ng_count": ng, "empty_count": empty}
            # Update timestamp to trigger frontend image refresh
            system_state["last_scan_time"] = time.time() 

            # 3. Hardware Sequence (Mock or Real)
            if not IS_WINDOWS:
                print("⚙️ Hardware Start...")
                # Logic from main.py adapted here if needed
                # For now keeping it simple as per your request to test inference first
                pass 
            
            system_state["sorting_active"] = False

        except Exception as e:
            print(f"❌ Scan Failed: {e}")
            system_state["sorting_active"] = False

    elif action == 'reset':
        print("\n↺ RESET")
        system_state["sorting_active"] = False
        system_state["cocoon_grid"] = [0] * 144
        system_state["stats"] = { "g_count": 0, "ng_count": 0, "empty_count": 0 }
        system_state["last_scan_time"] = None

    return jsonify({"status": "success"})

if __name__ == '__main__':
    # Ensure output dir exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    app.run(host='0.0.0.0', port=5000)

    
    
# API END POINTS
#  run server @ ip; main script connects to same IP to access API
#  