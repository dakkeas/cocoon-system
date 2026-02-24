from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import datetime

app = Flask(__name__)
CORS(app)

# --- Global state ---
system_state = {
    "sorting_active": False,
    "cocoon_grid": [0]*144, 
    "stats": {"g_count":0, "ng_count":0, "empty_count":0},
    "latest_log": "",
    "pending_command": None,
    "last_scan_id": None
} 
UPLOAD_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- HELPER: Translate Dictionary to Flat List ---
def process_grid_data(raw_data):
    """
    Safely converts dictionary keys into a flat list.
    0=Pending (White), 1=Good (Green), 2=NG (Red), 3=Empty (Grey)
    """
    flat_grid = []
    g_cnt = 0
    ng_cnt = 0
    empty_cnt = 0
    
    for r in range(1, 13):
        # robustly get row data (str or int key)
        row_items = raw_data.get(str(r))
        if row_items is None:
            row_items = raw_data.get(r)
        
        # If row is missing (not scanned yet), mark as Pending
        if not isinstance(row_items, list):
            row_items = ["Pending"] * 12
        
        for item in row_items:
            if item == "G":
                flat_grid.append(1) # Green
                g_cnt += 1
            elif item == "NG":
                flat_grid.append(2) # Red
                ng_cnt += 1
            elif item == "Empty":
                flat_grid.append(3) # Grey
                empty_cnt += 1
            else:
                flat_grid.append(0) # White/Pending
                
    return flat_grid, {"g_count": g_cnt, "ng_count": ng_cnt, "empty_count": empty_cnt}

@app.route('/api/upload_json', methods=['POST'])
def upload_json():
    try:
        data = request.get_json()
        if not data: return {"status":"fail"}, 400
        
        # --- NEW: Handle Row-by-Row Updates ---
        if "row_update" in data:
            row_dict = data["row_update"] # e.g. {"1": ["G", "NG", ...]}
            row_num = list(row_dict.keys())[0]
            row_items = row_dict[row_num]
            
            # Convert row to flat indices
            start_idx = (int(row_num) - 1) * 12
            for i, item in enumerate(row_items):
                val = 0
                if item == "G": val = 1
                elif item == "NG": val = 2
                elif item == "Empty": val = 3
                system_state['cocoon_grid'][start_idx + i] = val
            
            # Recalculate stats based on current grid
            g = system_state['cocoon_grid'].count(1)
            ng = system_state['cocoon_grid'].count(2)
            e = system_state['cocoon_grid'].count(3)
            system_state['stats'] = {"g_count": g, "ng_count": ng, "empty_count": e}
            
        # Handle full grid (original logic)
        elif "1" in data or 1 in data:
            grid, stats = process_grid_data(data)
            system_state['cocoon_grid'] = grid
            system_state['stats'] = stats
            
        return {"status":"success"}
    except Exception as e:
        print(f"JSON Error: {e}")
        return {"status":"error"}, 500

@app.route('/api/upload_log', methods=['POST'])
def upload_log():
    data = request.get_json()
    if data and "log" in data: system_state['latest_log'] = data["log"]
    return {"status":"success"}


@app.route("/api/upload_image_path", methods=["POST"])
def upload_image_path():
    data = request.get_json()
    path = data.get("path")
    print(f"INFERRED IMAGE PATH: {path}")
    
    if not path or not os.path.exists(path):
        return {"status": "file not found"}, 404

    # Update system state
    system_state["last_scan_id"] = path       # used internally
    system_state["latest_image_path"] = path  # this will be sent to frontend via /api/latest_json
    print("Updated system state with latest image path:", system_state["latest_image_path"])
    
    
    return {"status": "success"}


@app.route('/api/latest_json', methods=['GET'])
def latest_json():
    # Frontend fetches this
    return jsonify(system_state)

@app.route('/api/latest_log', methods=['GET'])
def latest_log():
    return jsonify({"log": system_state.get('latest_log', '')})

@app.route('/api/action', methods=['POST'])
def handle_action():
    # Frontend sends commands here
    data = request.get_json()
    action = data.get('action')
    system_state['pending_command'] = action # Save for main.py
    
    if action == 'start':
        system_state['latest_log'] = "▶ System Start Requested"
    elif action == 'stop':
        system_state['latest_log'] = "⏹ System Stop Requested"
    elif action == 'reset':
        system_state['cocoon_grid'] = [0] * 144
        system_state['stats'] = {"g_count":0, "ng_count":0, "empty_count":0}
        system_state['latest_log'] = "♻️ Batch Reset"
        
    return jsonify({"status": "success"})

@app.route('/api/get_command', methods=['GET'])
def get_command():
    # Main.py asks for commands here
    cmd = system_state.get('pending_command')
    system_state['pending_command'] = None 
    return jsonify({"command": cmd})


@app.route('/api/latest_image', methods=['GET'])
def latest_image():
    """
    Serve the latest image to React as actual file bytes.
    Uses the full path stored in system_state['last_scan_id'].
    """
    print('Received request for latest image. Checking path in system state...')
    path = system_state.get("last_scan_id")


    if not path:
        return {"error": "no image available"}, 404

    if not os.path.exists(path):
        return {"error": "file not found"}, 404

    print("SENDING FILE BYTES TO FRONTEND ......")
    # Send the image bytes directly
    return send_file(path, mimetype='image/jpeg', as_attachment=False)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)