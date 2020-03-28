#import RPi.GPIO as GPIO
#GPIO.setmode(GPIO.BCM) # GPIO Numbers instead of board numbers

#RELAIS_1_GPIO = 4
#GPIO.setup(RELAIS_1_GPIO, GPIO.OUT) # GPIO Assign mode
#GPIO.output(RELAIS_1_GPIO, GPIO.LOW) # out
#GPIO.output(RELAIS_1_GPIO, GPIO.HIGH) # on

import time
import RPi.GPIO as GPIO

RELAY_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

while 1:
    GPIO.output(RELAY_PIN, 1)

    time.sleep(1)

    GPIO.output(RELAY_PIN, 0)
    time.sleep(1)
GPIO.cleanup()
