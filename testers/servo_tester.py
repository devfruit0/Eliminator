from gpiozero import AngularServo
from time import sleep

servo = AngularServo(18, min_angle=-45, max_angle=45)

while True:
    servo.angle = -45
    sleep(2)
    servo.angle = -20
    sleep(2)
    servo.angle = 0
    sleep(2)
    servo.angle = 20
    sleep(2)
    servo.angle = 45
    sleep(2)

