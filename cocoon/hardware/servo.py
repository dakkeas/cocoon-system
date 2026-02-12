from adafruit_servokit import ServoKit
import threading
import time



'''
EXAMPLE USAGE:
if __name__ == "__main__":
    row = ['G', 'NG', 'G', 'NG', 'G', 'NG', 'G', 'NG', 'G', 'NG', 'G', 'NG']

    servos = ServoController()

    # Start the servo thread
    servos.start(row)

    # Let it run for 2 seconds
    time.sleep(2)

    # Stop all servos if needed
    servos.stop()
'''


class ServoController:
    def __init__(self, channels=12, pwm_freq=50):
        """
        channels: number of servo channels (default 12 for PCA9685)
        """
        self.kit = ServoKit(channels=channels)
        self.stop_event = threading.Event()
        self.servo_thread = None
        self.channels = channels

    def _activate_servos(self, row_data):
        """
        Internal method to move servos.
        Only moves servos where row_data[i] == 'G'.
        """
        if len(row_data) != self.channels:
            raise ValueError(f"row_data must contain exactly {self.channels} elements.")

        print("---------- activating servos ----------")

        try:
            # Move 'G' servos to 180°
            for ch in range(self.channels):
                if self.stop_event.is_set():
                    return
                if row_data[ch] == 'G':
                    self.kit.servo[ch].angle = 180

            time.sleep(0.5)

            # Move 'G' servos back to 0°
            for ch in range(self.channels):
                if self.stop_event.is_set():
                    return
                if row_data[ch] == 'G':
                    self.kit.servo[ch].angle = 0

            time.sleep(0.5)

            # Release 'G' servos
            for ch in range(self.channels):
                if row_data[ch] == 'G':
                    self.kit.servo[ch].angle = None

            time.sleep(0.1)

        except Exception as e:
            print("Error activating servos:", e)
        finally:
            print("Servo cycle complete.")

    def start(self, row_data):
        """Start servo activation in a separate thread."""
        self.stop_event.clear()
        self.servo_thread = threading.Thread(
            target=self._activate_servos,
            args=(row_data,),
            daemon=True
        )
        self.servo_thread.start()

    def stop(self):
        """Stop all servos immediately."""
        print("Stopping all servos...")
        self.stop_event.set()
        for ch in range(self.channels):
            self.kit.servo[ch].angle = None


