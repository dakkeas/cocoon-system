from cocoon.hardware import buttons
import time

buttons = buttons.ButtonController()

while True:
    buttons.wait_for_start()
    buttons.set_running(True)

    print("Main script running...")

    while not buttons.should_stop():
        time.sleep(1)

    print("Stopped.")
    buttons.reset()
    buttons.set_running(False)