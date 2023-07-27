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

def right_oriented(results):
    if results.multi_hand_landmarks is not None:
        x0, y0 = coordinate(results, 0, 0), coordinate(results, 0, 1)
        x9, y9 = coordinate(results, 9, 0), coordinate(results, 9, 1)
        if abs(x9 - x0) < 0.05:  
            m = 1000000000
        else:
            m = abs((y9 - y0)/(x9 - x0))   

        if m >= 0 and m <= 1 and x9 < x0:
            return True
    return False

def thumbs_up(results):
    if results.multi_hand_landmarks is not None:
        return right_oriented(results) and is_closed(results) and coordinate(results, 4, 1) < coordinate(results, 3, 1)
    return None

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    draw = False
    count = 0
    points = []

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
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        if thumbs_up(results):
            print("thumbs up")

        if is_closed(results):
            print('Closed')
            count += 1

            if count == 30:
                draw = not(draw)
                count = 0

        if results.multi_hand_landmarks is not None and draw:

            index_finger_coords = hand_landmarks.landmark[8]
            image_height, image_width, _ = image.shape
            x, y = int(index_finger_coords.x * image_width), int(index_finger_coords.y * image_height)
           
            points.append((x, y))
            #print(points)

            for i in range(len(points) - 1):
                cv2.line(image, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), color = (255, 255, 0), thickness = 10)            
        else:
            for i in range(len(points) - 1):
                cv2.line(image, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), color = (255, 255, 0), thickness = 10)            
            
        cv2.imshow('MediaPipe Hands', image)

        cv2.waitKey(1)
    
cap.release()

