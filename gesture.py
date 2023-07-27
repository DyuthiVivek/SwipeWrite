import cv2
import mediapipe as mp
import sys

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))
    
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='/home/dyuthi/SwipeWrite/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success,image = cap.read()
        image = cv2.flip(image,1)
        
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imageRGB)


        with GestureRecognizer.create_from_options(options) as recognizer:
            recognizer.recognize_async(mp_image, 5)


        cv2.imshow("Video",image)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()
