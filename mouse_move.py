import pyautogui as pag
import random
import time

combat_bt = {
    'tl' : {
        'x' : 771,
        'y' : 453
    },
    'br' : {
        'x' : 912,
        'y' : 542
    }
}

#while True:
    #x, y = pag.position()
    #print('X: %s, y: %s' % (x, y))
pag.moveTo(
    x=random.uniform(combat_bt['tl']['x'], combat_bt['br']['x']),
    y=random.uniform(combat_bt['tl']['y'], combat_bt['br']['y']),
    duration=random.uniform(0.5, 2)
)
    
pag.mouseDown()
time.sleep(random.uniform(0.15001, 1.23))
pag.mouseUp
