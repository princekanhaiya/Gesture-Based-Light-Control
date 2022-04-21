import cv2 as cv
import time
############################
cap_device, cap_width, cap_height=0, 640, 480
############################

# Camera Preparation ###############################################################
cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

pTime=0
while True:
    # Camera Capture #####################################################
    success, img=cap.read()
    if not success:
        break
    img= cv.flip(img, 1)  # Mirror view
  
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



