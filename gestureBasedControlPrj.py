import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import math
import handTrackingModule as htm
import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library

led_pin = 12           
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin, GPIO.OUT) 
pwm = GPIO.PWM(led_pin, 100)  
pwm.start(0)                    # Started PWM at 0% duty cycle


pwmRange = 100
minPwm = 0
maxPwm = 100
ledPwm = 0

def main():
    pTime=0

    # Camera Preparation ###############################################################
    cap_device, cap_width, cap_height=0, 640, 480
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    detector = htm.handDetector(maxHands=1,detectionCon=0.7)
    
    while True:
        # Camera Capture #####################################################
        success, img=cap.read()
        if not success:
            break
        img= cv.flip(img, 1)  # Mirror view
    
        img=detector.findHands(img)
        lmList = detector.findPosition(img,draw=False)
      
        if len(lmList) != 0:
            # print(lmList[4], lmList[8])
 
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
    
            length = math.hypot(x2 - x1, y2 - y1)
            # print(length)
    
            # Hand range 50 - 300
            # LED BRIGHTNESS Range  100 - 0
    
            ledPwm = np.interp(length, [50, 150], [minPwm, maxPwm])
            ledBar = np.interp(length, [50, 150], [400, 150])
            ledPwmPer = np.interp(length, [50, 150], [0, 100])

            print(int(length), ledPwm)
            pwm.ChangeDutyCycle(ledPwm)
    
            if length < 50:
                cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)
 
            cv.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv.rectangle(img, (50, int(ledBar)), (85, 400), (255, 0, 0), cv.FILLED)
            cv.putText(img, f'{int(ledPwmPer)} %', (40, 450), cv.FONT_HERSHEY_COMPLEX,1, (255, 0, 0), 3)

        #Frame rate
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime

        cv.putText(img, f'FPS:{int(fps)}', (10, 30),cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow("Hand Based Device Control",img)

        # Keying (ESC: Exit) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

if __name__ == "__main__":
    main()