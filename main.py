

import time
import sys
import logging
import signal
from typing import Dict, List

# --- Configuration & Tuning ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
ROW_COUNT = 12
SENSOR_POLL_RATE = 0.01  # 10ms delay to prevent CPU spinning

# Import Custom Modules
# Wrap in try/except to handle environment setup issues gracefully
try:
    from motor import Motor
    from servo import ServoController
    from sensor import PositionSensor
    from inference import VisionModel
except ImportError as e:
    print(f"CRITICAL ERROR: Missing module. {e}")
    sys.exit(1)

# Configure Logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("CocoonSorter")

class SystemController:
    def __init__(self):
        logger.info("Initializing System Components...")
        
        # Instantiate Hardware Modules
        try:
            self.motor = Motor()
            self.servo = ServoController()
            self.sensor = PositionSensor()
            self.vision = VisionModel()
        except Exception as e:
            logger.critical(f"Hardware initialization failed: {e}")
            sys.exit(1)

        self.running = True
        self.is_busy = False
        
        # Register Signal Handler for graceful exit (Ctrl+C)
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, sig, frame):
        """Handle KeyboardInterrupt safely."""
        logger.warning("Interrupt received. Shutting down...")
        self.shutdown()
        sys.exit(0)

    def health_check(self) -> bool:
        """Runs diagnostics on all subsystems."""
        logger.info("Running System Health Check...")
        
        try:
            # 1. Check Inference Model
            if not self.vision.is_loaded():
                raise RuntimeError("Vision model not loaded.")
            
            # 2. Check Sensor connectivity
            current_sensor_val = self.sensor.read()
            logger.info(f"Sensor initial state: {current_sensor_val}")

            # 3. Check Servo controller availability
            if not self.servo.check_connection():
                raise RuntimeError("Servo controller not responding.")

            # 4. Check Motor (Check if at start or can move)
            # Assuming motor.check_status() returns a boolean or dict
            if not self.motor.status_check():
                raise RuntimeError("Motor driver fault.")

            logger.info("✅ Health Check Passed.")
            return True

        except Exception as e:
            logger.error(f"❌ Health Check Failed: {e}")
            return False

    def ensure_home_position(self):
        """Ensures the motor is at the START position."""
        logger.info("Homing motor...")
        # Assuming motor.is_at_start() is available, otherwise move reverse
        if not self.motor.is_at_start():
            self.motor.move_to_start()
            # Wait/Poll until arrival (if move_to_start is non-blocking)
            while not self.motor.is_at_start():
                time.sleep(0.1)
        logger.info("Motor at START position.")

    def process_row_movement(self, row_index: int):
        """
        Handles the alternating sensor logic to advance one row.
        Logic: Wait for current state -> Move -> Wait for state change -> Stop.
        """
        # Determine expected state based on row parity
        # Even rows (0, 2...) might be HIGH, Odd (1, 3...) LOW, or vice versa.
        # Logic: We move WHILE the sensor is in the 'previous' state until it flips.
        
        start_state = self.sensor.read()
        target_state = not start_state # We stop when state flips
        
        logger.debug(f"Row {row_index}: Moving (Wait for {start_state} -> {target_state})")
        
        self.motor.move_forward()
        
        # Safety timeout variables could be added here
        while self.sensor.read() != target_state:
            time.sleep(SENSOR_POLL_RATE)
            
        self.motor.stop()
        logger.info(f"Row {row_index} position reached.")

    def run_sorting_cycle(self):
        """Executes the full 12-row sorting workflow."""
        if self.is_busy:
            logger.warning("System is already busy!")
            return

        self.is_busy = True
        logger.info("📢 Starting Sorting Cycle")

        try:
            # Step A: Run Inference
            logger.info("📸 Capturing Vision Grid...")
            grid_results: Dict[int, List[str]] = self.vision.run_inference()
            
            if len(grid_results) != ROW_COUNT:
                raise ValueError(f"Inference returned {len(grid_results)} rows, expected {ROW_COUNT}")

            # Step B: Ensure Start Position
            self.ensure_home_position()

            # Step C: Process Rows 0 to 11
            for row_idx in range(ROW_COUNT):
                logger.info(f"--- Processing Row {row_idx} ---")
                
                # 1. Actuate Servos for this row
                row_data = grid_results.get(row_idx, [])
                self.servo.sort_row(row_data)
                
                # 2. Wait briefly for servos to settle (optional, configurable)
                time.sleep(0.2)
                
                # 3. Advance to next row (skip movement after the very last sort if desired, 
                # but usually we need to clear the machine)
                self.process_row_movement(row_idx)

            # Step D: End of Grid Behavior
            logger.info("Grid complete. Moving to END position...")
            self.motor.move_to_end() 
            # Wait for end limit switch
            while not self.motor.is_at_end():
                time.sleep(0.1)
                
            logger.info("Returning to HOME...")
            self.ensure_home_position()
            
            logger.info("✅ Cycle Complete. Ready.")

        except Exception as e:
            logger.error(f"⚠️ Cycle Aborted: {e}")
            self.motor.stop() # Emergency stop
        
        finally:
            self.is_busy = False

    def shutdown(self):
        """Safe shutdown sequence."""
        logger.info("Shutting down system...")
        self.motor.stop()
        self.servo.reset_all() # Optional: move servos to neutral
        logger.info("System Halted.")

    def cmd_loop(self):
        """Blocking command listener loop."""
        print("\n🤖 System Ready. Commands: start | status | home | exit")
        
        while self.running:
            try:
                cmd = input(">> ").strip().lower()
                
                if cmd == "exit":
                    self.shutdown()
                    self.running = False
                    break
                
                elif cmd == "start":
                    # Run in main thread for now (simplifies logic), 
                    # but could be threaded if async commands are needed during sort.
                    self.run_sorting_cycle()
                    
                elif cmd == "home":
                    if not self.is_busy:
                        self.ensure_home_position()
                    else:
                        logger.warning("Cannot home while busy.")
                        
                elif cmd == "status":
                    state = "BUSY" if self.is_busy else "IDLE"
                    print(f"System State: {state}")
                    print(f"Sensor State: {self.sensor.read()}")
                    
                elif cmd == "":
                    continue
                else:
                    print("Unknown command.")
                    
            except KeyboardInterrupt:
                self._handle_interrupt(None, None)
            except Exception as e:
                logger.error(f"Command Loop Error: {e}")

if __name__ == "__main__":
    # Bootstrap
    sys_ctrl = SystemController()
    
    if sys_ctrl.health_check():
        sys_ctrl.cmd_loop()
    else:
        logger.critical("System failed health checks. Exiting.")
        sys.exit(1)
