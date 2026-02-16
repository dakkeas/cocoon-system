import RPi.GPIO as GPIO
import logging

class MotorSystem:
    def __init__(self, config, pwm_freq=1000):
        """
        Controls multiple L298N drivers wired in parallel (or independent) 
        as a single logical motor unit.
        
        config: Dictionary containing all pin definitions.
        """
        self.logger = logging.getLogger("Motor")
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False) # Suppress warnings about pins being already in use

        # 1. Extract all pin groups from config
        # We group them into 'A' side (Forward=High) and 'B' side (Forward=Low)
        self.pins_fwd_high = [] # Pins that go HIGH for forward (IN1, IN3...)
        self.pins_fwd_low  = [] # Pins that go LOW  for forward (IN2, IN4...)
        self.pwm_pins = set()   # Unique set of Enable pins
        
        # Parse Config and sort pins
        self._parse_config(config)

        # 2. Setup GPIO Direction Pins
        all_dir_pins = set(self.pins_fwd_high + self.pins_fwd_low)
        for pin in all_dir_pins:
            GPIO.setup(pin, GPIO.OUT)

        # 3. Setup PWM (Enable) Pins
        # We use a dictionary to store PWM instances: { pin_number: pwm_object }
        # This prevents trying to create two PWMs on the same pin (Pin 17).
        self.pwms = {} 
        
        for pin in self.pwm_pins:
            GPIO.setup(pin, GPIO.OUT)
            pwm_instance = GPIO.PWM(pin, pwm_freq)
            pwm_instance.start(0)
            self.pwms[pin] = pwm_instance
            
        self.logger.info(f"Motor initialized with {len(self.pwms)} PWM channels.")

    def _parse_config(self, cfg):
        """Helper to unpack the variables into lists."""
        # We manually map the provided config variables to logic
        # IN1/IN3 are usually the "Left" side of the H-Bridge
        # IN2/IN4 are usually the "Right" side of the H-Bridge
        
        # Driver 1
        self.pins_fwd_high.extend([cfg['D1_IN1'], cfg['D1_IN3']])
        self.pins_fwd_low.extend([cfg['D1_IN2'], cfg['D1_IN4']])
        self.pwm_pins.update([cfg['D1_ENA'], cfg['D1_ENB']])

        # Driver 2
        self.pins_fwd_high.extend([cfg['D2_IN1'], cfg['D2_IN3']])
        self.pins_fwd_low.extend([cfg['D2_IN2'], cfg['D2_IN4']])
        self.pwm_pins.update([cfg['D2_ENA'], cfg['D2_ENB']])

    def forward(self, speed=100):
        """Move all connected drivers forward."""
        # Set High Side
        GPIO.output(self.pins_fwd_high, GPIO.HIGH)
        # Set Low Side
        GPIO.output(self.pins_fwd_low, GPIO.LOW)
        # Set Speed
        self._set_pwm(speed)

    def backward(self, speed=100):
        """Move all connected drivers backward."""
        # Flip logic
        GPIO.output(self.pins_fwd_high, GPIO.LOW)
        GPIO.output(self.pins_fwd_low, GPIO.HIGH)
        # Set Speed
        self._set_pwm(speed)

    def stop(self):
        """Stop all drivers."""
        GPIO.output(self.pins_fwd_high, GPIO.LOW)
        GPIO.output(self.pins_fwd_low, GPIO.LOW)
        self._set_pwm(0)

    def _set_pwm(self, speed):
        """Internal helper to update all PWM instances."""
        for pwm in self.pwms.values():
            pwm.ChangeDutyCycle(speed)

    def cleanup(self):
        """Stops PWM and cleans up GPIO."""
        self.stop()
        for pwm in self.pwms.values():
            pwm.stop()
        GPIO.cleanup()