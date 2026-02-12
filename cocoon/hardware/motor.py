import RPi.GPIO as GPIO

'''
EXAMPLE USE IN main.py

# Set BCM mode
GPIO.setmode(GPIO.BCM)

# Create motor objects with BCM pin numbers
driver1 = MotorDriver(in1=27, in2=22, en=17)
driver2 = MotorDriver(in1=23, in2=24, en=25)

# Use the motors
driver1.forward(70)
driver2.forward(70)

# Stop
driver1.stop()
driver2.stop()

# Cleanup GPIO at the end
GPIO.cleanup()
'''

class MotorDriver:
    def __init__(self, in1, in2, en, pwm_freq=1000):
        """
        in1, in2 = direction pins
        en = enable pin (PWM)
        pwm_freq = PWM frequency
        """
        self.in1 = in1
        self.in2 = in2
        self.en = en
        
        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        GPIO.setup(self.en, GPIO.OUT)
        

        self.pwm = GPIO.PWM(self.en, pwm_freq)
        self.pwm.start(0)

    def forward(self, speed=100):
        """
        Move motor forward.
        speed: 0–100
        """
        GPIO.output(self.in1, GPIO.HIGH)
        GPIO.output(self.in2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(speed)

    def backward(self, speed=100):
        """
        Move motor backward.
        speed: 0–100
        """
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.HIGH)
        self.pwm.ChangeDutyCycle(speed)

    def set_speed(self, speed):
        """
        Change speed without changing direction.
        """
        self.pwm.ChangeDutyCycle(speed)

    def stop(self):
        """
        Stop motor completely.
        """
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(0)

    def cleanup(self):
        self.pwm.stop()



