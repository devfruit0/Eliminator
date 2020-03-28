import gpiozero
import time

RELAY_PIN = 17
relay = gpiozero.OutputDevice(RELAY_PIN)

while True:
    print(relay.value)
    relay.on()
    time.sleep(.5)
    print(relay.value)
    relay.off()
    time.sleep(.5)
