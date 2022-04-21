import copy
import argparse

import cv2 as cv
from cv2 import sqrt
import numpy as np
import mediapipe as mp
import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from time import sleep # Import the sleep function from the time module
from utils import CvFpsCalc

from PyQt5 import QtCore, QtGui, QtWidgets  #PyQt library for GUI
from guiBarExample import PowerBar  #Library for LED GUI

led_pin = 12           
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin, GPIO.OUT) 
pwm = GPIO.PWM(led_pin, 100)  
pwm.start(0)                    # Started PWM at 0% duty cycle

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence",help='min_detection_confidence',type=float,default=0.7)
    parser.add_argument("--min_tracking_confidence",help='min_tracking_confidence',type=int,default=0.5)
    parser.add_argument('--use_brect', action='store_true')
    parser.add_argument('--plot_world_landmark', action='store_true')

    args = parser.parse_args()

    return args

def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # Wrist 1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # Wrist 2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # index finger: base
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # Middle finger: Base
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # Ring finger: base
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # Little finger: Base
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_landmarks(image, cx, cy, landmarks, handedness):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    ax=0
    ay=0
    qx=0
    qy=0

    # Key
    for index, landmark in enumerate(landmarks.landmark):        
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z
        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # Wrist 1
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # Wrist 2
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # Thumb: Base
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # Thumb: First joint
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # Thumb: Fingertips
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (255, 0, 0), 2)
            ax=landmark_x
            ay=landmark_y
        if index == 5:  # index finger: base
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # Index finger: Second joint
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # Index finger: First joint
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # Index finger: Fingertips
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (255, 0, 0), 2)
            qx=landmark_x
            qy=landmark_y
        if index == 9:  # Middle finger: Base
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # Middle finger: Second joint
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # Middle finger: First joint
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # Middle finger: Refers first
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # Ring finger: base
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # Ring finger: Second joint
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # Ring finger: First joint
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # Ring finger: fingertips
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # Little finger: Base
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # Little finger: Second joint
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # Little finger: First joint
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # Little finger: Fingertips
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

        n=(int)((ax-qx)**2+(ay-qy)**2)**(0.5)
        #print(n)
        translate(n, 20, 200, 99, 0)
    # Connecting line
    if len(landmark_point) > 0:
        # thumb
        cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

        # little finger
        cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

        # middle finger
        cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

        # ring finger
        cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

        # little finger
        cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

        # Palm
        cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # Center of gravity + left and right
    if len(landmark_point) > 0:
        # handedness.classification[0].index
        # handedness.classification[0].score

        cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv.putText(image, handedness.classification[0].label[0],
                   (cx - 6, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                   2, cv.LINE_AA)  # label[0]:一Only the first letter

    return image

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    out_val=(int)(rightMin + (valueScaled * rightSpan))
    if out_val >98:
        out_val=99
    if out_val<0:
        out_val=0  
    pwm.ChangeDutyCycle(out_val)
    print(out_val)

def plot_world_landmarks(
    plt,
    ax_list,
    multi_hands_landmarks,
    multi_handedness,
    visibility_th=0.5,):
    ax_list[0].cla()
    ax_list[0].set_xlim3d(-0.1, 0.1)
    ax_list[0].set_ylim3d(-0.1, 0.1)
    ax_list[0].set_zlim3d(-0.1, 0.1)
    ax_list[1].cla()
    ax_list[1].set_xlim3d(-0.1, 0.1)
    ax_list[1].set_ylim3d(-0.1, 0.1)
    ax_list[1].set_zlim3d(-0.1, 0.1)

    for landmarks, handedness in zip(multi_hands_landmarks, multi_handedness):
        handedness_index = 0
        if handedness.classification[0].label == 'Left':
            handedness_index = 0
        elif handedness.classification[0].label == 'Right':
            handedness_index = 1

        landmark_point = []

        for index, landmark in enumerate(landmarks.landmark):
            landmark_point.append(
                [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

        palm_list = [0, 1, 5, 9, 13, 17, 0]
        thumb_list = [1, 2, 3, 4]
        index_finger_list = [5, 6, 7, 8]
        middle_finger_list = [9, 10, 11, 12]
        ring_finger_list = [13, 14, 15, 16]
        pinky_list = [17, 18, 19, 20]

        # Palm
        palm_x, palm_y, palm_z = [], [], []
        for index in palm_list:
            point = landmark_point[index][1]
            palm_x.append(point[0])
            palm_y.append(point[2])
            palm_z.append(point[1] * (-1))

        # thumb
        thumb_x, thumb_y, thumb_z = [], [], []
        for index in thumb_list:
            point = landmark_point[index][1]
            thumb_x.append(point[0])
            thumb_y.append(point[2])
            thumb_z.append(point[1] * (-1))

        # index finger
        index_finger_x, index_finger_y, index_finger_z = [], [], []
        for index in index_finger_list:
            point = landmark_point[index][1]
            index_finger_x.append(point[0])
            index_finger_y.append(point[2])
            index_finger_z.append(point[1] * (-1))

        # middle finger
        middle_finger_x, middle_finger_y, middle_finger_z = [], [], []
        for index in middle_finger_list:
            point = landmark_point[index][1]
            middle_finger_x.append(point[0])
            middle_finger_y.append(point[2])
            middle_finger_z.append(point[1] * (-1))

        # ring finger
        ring_finger_x, ring_finger_y, ring_finger_z = [], [], []
        for index in ring_finger_list:
            point = landmark_point[index][1]
            ring_finger_x.append(point[0])
            ring_finger_y.append(point[2])
            ring_finger_z.append(point[1] * (-1))

        # little finger
        pinky_x, pinky_y, pinky_z = [], [], []
        for index in pinky_list:
            point = landmark_point[index][1]
            pinky_x.append(point[0])
            pinky_y.append(point[2])
            pinky_z.append(point[1] * (-1))

        ax_list[handedness_index].plot(palm_x, palm_y, palm_z)
        ax_list[handedness_index].plot(thumb_x, thumb_y, thumb_z)
        ax_list[handedness_index].plot(index_finger_x, index_finger_y,
                                       index_finger_z)
        ax_list[handedness_index].plot(middle_finger_x, middle_finger_y,
                                       middle_finger_z)
        ax_list[handedness_index].plot(ring_finger_x, ring_finger_y,
                                       ring_finger_z)
        ax_list[handedness_index].plot(pinky_x, pinky_y, pinky_z)

    plt.pause(.001)

    return

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # The outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),(0, 255, 0), 2)

    return image

def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # model_complexity = args.model_complexity

    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect
    plot_world_landmark = args.plot_world_landmark

    # Camera Preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=max_num_hands,min_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence)

    # FPS Measurement Module ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # World Coordinate plot ########################################################
    if plot_world_landmark:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        r_ax = fig.add_subplot(121, projection="3d")
        l_ax = fig.add_subplot(122, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    while True:
        display_fps = cvFpsCalc.get()

        # Camera Capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror view
        debug_image = copy.deepcopy(image)

        # Detect execution #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        # drawing ################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Palm center of gravity calculation
                cx, cy = calc_palm_moment(debug_image, hand_landmarks)
                # Calculate circumscribed rectangles
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # drawing
                debug_image = draw_landmarks(debug_image, cx, cy,
                                             hand_landmarks, handedness)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # World coordinate plot ###################################################
        if plot_world_landmark:
            if results.multi_hand_world_landmarks is not None:
                plot_world_landmarks(
                    plt,
                    [r_ax, l_ax],
                    results.multi_hand_world_landmarks,
                    results.multi_handedness,
                )

        # Keying (ESC: Exit) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # Screen reflection #############################################################
        cv.imshow('MediaPipe Hand Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()