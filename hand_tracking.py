import cv2
import mediapipe as mp

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image
    
    def positionFinder(self,image, handNo=0, draw=True, finger_no=8):
        lmlist = {}
        if self.results.multi_hand_landmarks:
            x_= str(self.results.multi_hand_landmarks[-1].landmark[8]).split('\n')[0]
            y_= str(self.results.multi_hand_landmarks[-1].landmark[8]).split('\n')[1]
            z_= str(self.results.multi_hand_landmarks[-1].landmark[8]).split('\n')[2]

            x = float(x_.split(" ")[1])
            y = float(y_.split(" ")[1])
            z = float(z_.split(" ")[1])

            height = image.shape[0]
            width = image.shape[1]
            x_real = x * width
            y_real = y * height

            print(x_real, y_real)

            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist[id] = (cx,cy)
            if draw:
                cv2.circle(image,lmlist[finger_no], 15 , (255,0,255), cv2.FILLED)

        return lmlist
    
def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while cap.isOpened():
        success,image = cap.read()
        image = cv2.flip(image,1)
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        if len(lmList) != 0:
            print(lmList[4])
        cv2.imshow("Video",image)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()