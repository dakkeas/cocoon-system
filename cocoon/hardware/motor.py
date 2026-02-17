import RPi.GPIO as GPIO
import logging
import config

class MotorSystem:
    def __init__(self, config, pwm_freq=1000):
        """
        Simple 2x L298N motor system
        Driver 1: IN1-4, ENA, ENB
        Driver 2: IN1-4, ENA, ENB
        """

        self.set_direction = 'FORWARD'

        self.logger = logging.getLogger("Motor")

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # =========================
        # DRIVER 1 PINS
        # =========================
        self.d1_in1 = config.D1_IN1
        self.d1_in2 = config.D1_IN2
        self.d1_in3 = config.D1_IN3
        self.d1_in4 = config.D1_IN4
        self.d1_ena = config.D1_ENA
        self.d1_enb = config.D1_ENB

        # =========================
        # DRIVER 2 PINS
        # =========================
        self.d2_in1 = config.D2_IN1
        self.d2_in2 = config.D2_IN2
        self.d2_in3 = config.D2_IN3
        self.d2_in4 = config.D2_IN4
        self.d2_ena = config.D2_ENA
        self.d2_enb = config.D2_ENB

        # Setup direction pins
        self.dir_pins = [
            self.d1_in1, self.d1_in2, self.d1_in3, self.d1_in4,
            self.d2_in1, self.d2_in2, self.d2_in3, self.d2_in4
        ]

        for pin in self.dir_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        # Setup PWM pins
        self.pwm_pins = [
            self.d1_ena, self.d1_enb,
            self.d2_ena, self.d2_enb
        ]

        self.pwms = []

        for pin in self.pwm_pins:
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, pwm_freq)
            pwm.start(0)
            self.pwms.append(pwm)

        self.logger.info("MotorSystem initialized (2 drivers, 4 motors).")

    # =========================================
    # MOVEMENT FUNCTIONS
    # =========================================

    def forward(self, speed=100):
        """All 4 motors forward"""

        # Driver 1
        GPIO.output(self.d1_in1, GPIO.HIGH)
        GPIO.output(self.d1_in2, GPIO.LOW)
        GPIO.output(self.d1_in3, GPIO.HIGH)
        GPIO.output(self.d1_in4, GPIO.LOW)

        # Driver 2
        GPIO.output(self.d2_in1, GPIO.HIGH)
        GPIO.output(self.d2_in2, GPIO.LOW)
        GPIO.output(self.d2_in3, GPIO.HIGH)
        GPIO.output(self.d2_in4, GPIO.LOW)

        self._set_speed(speed)

    def backward(self, speed=100):
        """All 4 motors backward"""

        # Driver 1
        GPIO.output(self.d1_in1, GPIO.LOW)
        GPIO.output(self.d1_in2, GPIO.HIGH)
        GPIO.output(self.d1_in3, GPIO.LOW)
        GPIO.output(self.d1_in4, GPIO.HIGH)

        # Driver 2
        GPIO.output(self.d2_in1, GPIO.LOW)
        GPIO.output(self.d2_in2, GPIO.HIGH)
        GPIO.output(self.d2_in3, GPIO.LOW)
        GPIO.output(self.d2_in4, GPIO.HIGH)

        self._set_speed(speed)

    def stop(self):
        """Stop all motors"""

        for pin in self.dir_pins:
            GPIO.output(pin, GPIO.LOW)

        self._set_speed(0)


    def start(self, speed):

        if self.set_direction == 'FORWARD':
            self.forward(speed)
        elif self.set_direction == 'BACKWARD':
            self.backward(speed)

    # =========================================
    # INTERNAL
    # =========================================

    def _set_speed(self, speed):
        for pwm in self.pwms:
            pwm.ChangeDutyCycle(speed)

    def cleanup(self):
        self.stop()
        for pwm in self.pwms:
            pwm.stop()
        GPIO.cleanup()
