import mediapipe as mp
import cv2
import gesture

class HandTracking:

    def __init__(self, image) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.draw = False
        self.count = 0
        self.new_points = []
        self.all_points = []
        self.image = image
    
    def read_video_capture(self) -> None:
        self.image = cv2.cvtColor(cv2.flip(self.image, 1), cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.image)
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        return True
    
    def draw_hand_connections(self) -> None:
        if self.results.multi_hand_landmarks:                      
            for self.hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(self.image, self.hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    
    def update_draw(self) -> None:
        if gesture.is_closed(self.results):
            self.count += 1

            if self.count == 30:
                self.draw = not(self.draw)
                self.count = 0

    def actual_coordinates(self) -> None:
        index_finger_coords = self.hand_landmarks.landmark[8]
        image_height, image_width, _ = self.image.shape
        self.x, self.y = index_finger_coords.x, index_finger_coords.y
  
    def draw_line(self) -> None:
        if self.results.multi_hand_landmarks is not None and self.draw:
            self.actual_coordinates()
            self.new_points.append((self.x, self.y))

            for points in self.all_points:
                for i in range(len(points) - 1):
                    cv2.line(self.image, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), color = (255, 255, 0), thickness = 10)       
            for i in range(len(self.new_points) - 1):
                    cv2.line(self.image, (self.new_points[i][0], self.new_points[i][1]), (self.new_points[i + 1][0], self.new_points[i + 1][1]), color = (255, 255, 0), thickness = 10)                 
        else:
            for points in self.all_points:
                for i in range(len(points) - 1):
                    cv2.line(self.image, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), color = (255, 255, 0), thickness = 10)  
            self.all_points.append(self.new_points)
            self.new_points = []          
        
    def start_drawing(self) -> None:
        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as self.hands:
            if not(self.read_video_capture()):
                return None
            self.draw_hand_connections()

            if self.results.multi_hand_landmarks is not None:
                self.actual_coordinates()
                return gesture.is_closed(self.results), self.x, self.y, self.image
            
            return False, 0, 0, self.image

