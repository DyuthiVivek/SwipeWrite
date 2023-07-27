import mediapipe as mp
import cv2
import numpy as np
from math import dist

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def coordinate(results, landmark, num):
    return float(str(results.multi_hand_landmarks[-1].landmark[landmark]).split('\n')[num].split(" ")[1])

def is_closed(results):
    if results.multi_hand_landmarks is not None:
        try:
            p0x, p0y = coordinate(results, 0, 0), coordinate(results, 0, 1)

            p7x, p7y = coordinate(results, 7, 0), coordinate(results, 7, 1)
            d07 = dist([p0x, p0y], [p7x, p7y])

            p8x, p8y = coordinate(results, 8, 0), coordinate(results, 8, 1)
            d08 = dist([p0x, p0y], [p8x, p8y])

            p11x, p11y = coordinate(results, 11, 0), coordinate(results, 11, 1)
            d011 = dist([p0x, p0y], [p11x, p11y])

            p12x, p12y = coordinate(results, 12, 0), coordinate(results, 12, 1)
            d012 = dist([p0x, p0y], [p12x, p12y])

            p15x, p15y = coordinate(results, 15, 0), coordinate(results, 15, 1)
            d015 = dist([p0x, p0y], [p15x, p15y])

            p16x, p16y = coordinate(results, 16, 0), coordinate(results, 16, 1)
            d016 = dist([p0x, p0y], [p16x, p16y])

            p19x, p19y = coordinate(results, 19, 0), coordinate(results, 19, 1)
            d019 = dist([p0x, p0y], [p19x, p19y])

            p20x, p20y = coordinate(results, 20, 0), coordinate(results, 20, 1)
            d020 = dist([p0x, p0y], [p20x, p20y])

            return d07 > d08 and d011 > d012 and d015 > d016 and d019 > d020
                    
        except:
           pass

    
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    draw = False
    count = 0
    while cap.isOpened():
        success, image = cap.read()
        image = image.copy()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:                      
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # if results.multi_handedness != None:
        #     for idx, hand_handedness in enumerate(results.multi_handedness):
        #         print(hand_handedness.classification[0].label)

        # if is_closed(results):
        #     print("Closed")
        # else:
        #     print("Open")

        if is_closed(results):
            print('Closed')
            count += 1

            if count == 30:
                draw = not(draw)
                count = 0

        if results.multi_hand_landmarks is not None and draw:

            points = []
            plandmark_x, plandmark_y = coordinate(results, 8, 0), coordinate(results, 8, 1)    
            # index_finger_coords = hand_landmarks.landmark[8]
            # image_height, image_width, _ = image.shape
            # x, y = int(index_finger_coords.x * image_width), int(index_finger_coords.y * image_height)
            # cv2.circle(image, (x, y), 10, (0, 255, 0), -1)


            points.append((int(640*plandmark_x), int(450*plandmark_y)))
            #since landmark 8 is the tip of index finger

            print(points)

            cv2.line(image, (points[0][0], points[0][1]), (points[0][0], points[0][1]), color = (255, 255, 0), thickness = 25)            
            #cv2.line(image, (100, 100), (300, 300), color = (255, 255, 0), thickness = 10)            

        cv2.imshow('MediaPipe Hands', image)

        cv2.waitKey(1)
    
cap.release()

