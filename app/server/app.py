import sys
import os
import platform
import json
import time
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

IS_WINDOWS = platform.system() == "Windows"

# --- IMPORTS ---
try:
    import config
    from utils.inference import VisionSystem
    if not IS_WINDOWS:
        from hardware.motor import MotorDriver
        from hardware.servo import ServoController
        from hardware.sensor import IR_Sensor
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# --- GLOBAL STATE ---
system_state = {
    "sorting_active": False,
    "cocoon_grid": [0] * 144, 
    "stats": { "g_count": 0, "ng_count": 0, "empty_count": 0 },
    "matrix_view": {},
    "coordinate_map": {}
}

# --- INITIALIZATION ---
vision = VisionSystem(model_name=config.MODEL_NAME, model_dir=os.path.join(project_root, "models"), camera_index=-1 if IS_WINDOWS else config.CAMERA_INDEX)
servos = None if IS_WINDOWS else ServoController()
motor = None if IS_WINDOWS else MotorDriver(config.MOTOR_DRIVER_1_in1, config.MOTOR_DRIVER_1_in2, config.MOTOR_DRIVER_1_en1)
ir_sensor = None if IS_WINDOWS else IR_Sensor()

# --- API ROUTES ---

@app.route('/api/data', methods=['GET'])
def get_data():
    """FIXED: This was the missing route causing the 404 errors!"""
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
    data = request.get_json()
    action = data.get('action')
    
    if action == 'start':
        print(f"\n▶ {'[TEST] ' if IS_WINDOWS else ''}Scanning Request Received...")
        system_state["sorting_active"] = True
        
        try:
            # 1. AI Classification
            grid_dict = vision.run_inference() 
            
            # 2. Build Maps and Stats
            matrix_view, coord_map, flat_grid = {}, {}, []
            g, ng, empty = 0, 0, 0
            
            for r in range(1, 13):
                row_data = grid_dict[r]
                matrix_view[f"Row {r:02}"] = " ".join([f"[{item:^5}]" for item in row_data])
                for c in range(1, 13):
                    val = row_data[c-1]
                    coord_map[f"[{r},{c}]"] = val
                    flat_grid.append(1 if val == "G" else 2 if val == "NG" else 3)
                
                g += row_data.count("G"); ng += row_data.count("NG"); empty += row_data.count("Empty")

            # Update State
            system_state.update({
                "cocoon_grid": flat_grid,
                "matrix_view": matrix_view, 
                "coordinate_map": coord_map
            })
            system_state["stats"].update({"g_count": g, "ng_count": ng, "empty_count": empty})

            # 3. Save Timestamped JSON
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scan_{ts}.json"
            with open(filename, 'w') as f:
                json.dump(system_state, f, indent=4)
            print(f"💾 Results saved to {filename}")

            # 4. Sequential Hardware Workflow (Only on Raspberry Pi)
            if not IS_WINDOWS:
                print("⚙️ Initiating Sequential Row Sorting...")
                for r in range(1, 13):
                    print(f"--- Row {r:02} ---")
                    # Move motor forward until white edge detected
                    motor.forward(speed=40)
                    while ir_sensor.read() == 0: # Wait for white reflection
                        time.sleep(0.01)
                    motor.stop()
                    
                    # Fire actuators based on classification for this row
                    print(f"Pushing out Good cocoons in Row {r}...")
                    servos.start(grid_dict[r])
                    time.sleep(0.8) # Wait for mechanical ejection
                
                # Reverse Home
                print("🔄 Sorting Finished. Returning to Row 1...")
                motor.backward(speed=70)
                time.sleep(5.0) # Adjust based on physical length
                motor.stop()
            
            system_state["sorting_active"] = False

        except Exception as e:
            print(f"❌ Scan Failed: {e}")
            system_state["sorting_active"] = False

    elif action == 'reset':
        print("\n↺ [RESET] System Cleared")
        system_state["sorting_active"] = False
        system_state["cocoon_grid"] = [0] * 144
        system_state["stats"] = { "g_count": 0, "ng_count": 0, "empty_count": 0 }
        system_state["matrix_view"], system_state["coordinate_map"] = {}, {}

    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)