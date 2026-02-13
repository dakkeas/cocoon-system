import sys
import os
import time
# Note: 'threading' is no longer needed since we aren't looping in the background
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- DEBUG PRINT ---
print(f"📂 Project Root detected at: {project_root}")

# --- IMPORTS ---
try:
    from hardware.motor import MotorDriver
    from hardware.servo import ServoController
    from hardware.sensor import IR_Sensor
    from utils.inference import VisionSystem
    import config
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}\n")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# --- HARDWARE INITIALIZATION ---
motor1 = None
vision = None
servos = None
sensor = None

try:
    print("Initializing Hardware...")
    models_folder = os.path.join(project_root, "models")
    print(f"🔍 Looking for model in: {models_folder}")

    vision = VisionSystem(
        model_name=config.MODEL_NAME, 
        model_dir=models_folder, 
        camera_index=config.CAMERA_INDEX
    )
    
    servos = ServoController()
    sensor = IR_Sensor()
    
    # Motor is initialized but won't move until you tell it to
    motor1 = MotorDriver(config.MOTOR_DRIVER_1_in1, config.MOTOR_DRIVER_1_in2, config.MOTOR_DRIVER_1_en1)
    
    print("✅ Hardware initialized successfully.")

except Exception as e:
    print(f"❌ Hardware Init Failed: {e}")

# --- GLOBAL STATE ---
system_state = {
    "sorting_active": False,
    "cocoon_grid": [0] * 144, 
    "stats": { "g_count": 0, "ng_count": 0, "empty_count": 0 }
}

# --- API ROUTES ---

@app.route('/api/data')
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
        "active": system_state["sorting_active"]
    })

@app.route('/api/action', methods=['POST'])
def handle_action():
    action = request.json.get('action')
    
    if action == 'start':
        print("▶ Scanning Request Received...")
        system_state["sorting_active"] = True
        
        # --- SINGLE SHOT LOGIC STARTS HERE ---
        try:
            # 1. Capture 1 frame & Classify
            grid_dict = vision.run_inference() 
            
            # 2. Flatten result into a list for the React Grid
            flat_grid = []
            g = 0; ng = 0; empty = 0
            
            for row in range(1, 13):
                if row in grid_dict:
                    row_data = grid_dict[row]
                    
                    # (Optional) Trigger servos here if needed
                    # servos.start(row_data)

                    flat_grid.extend([
                        1 if x == "G" else 2 if x == "NG" else 3 
                        for x in row_data
                    ])
                    g += row_data.count("G")
                    ng += row_data.count("NG")
                    empty += row_data.count("Empty")
                else:
                    flat_grid.extend([0] * 12)

            # 3. Update Global State
            system_state["cocoon_grid"] = flat_grid
            system_state["stats"]["g_count"] = g
            system_state["stats"]["ng_count"] = ng
            system_state["stats"]["empty_count"] = empty
            
            print("✅ Scan Complete. Website updated.")

        except Exception as e:
            print(f"❌ Scan Failed: {e}")
        # --- SINGLE SHOT LOGIC ENDS HERE ---

    elif action == 'stop':
        system_state["sorting_active"] = False
        print("⏹ System STOPPED")

    elif action == 'reset':
        system_state["sorting_active"] = False
        system_state["cocoon_grid"] = [0] * 144
        system_state["stats"] = { "g_count": 0, "ng_count": 0, "empty_count": 0 }
        print("↺ System RESET")

    return jsonify({"status": "success", "active": system_state["sorting_active"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)