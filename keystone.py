import pyautogui as pag
import keyboard
import time

resource = {'replay.png','start.png','vote.png','end.png','gas.png','deck.png','money.png'}
print("start")
while True:
    check = 0
    for i in resource:
        temp = temp = pag.locateCenterOnScreen(i, confidence = 0.95)
        if temp:
            pag.click(temp)
            time.sleep(2)
            if i == 'money.png':
                pag.click(temp)
                time.sleep(4)
                pag.press("F10")
                time.sleep(2)
                pag.press("Q")
                temp = pag.locateCenterOnScreen('quit.png', confidence = 0.95)
                pag.click(temp)
                
        if keyboard.is_pressed("P"):
            print("end")
            exit()
            check=1;

    if(check == 1):
        break;