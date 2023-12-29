from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
from io import BytesIO
from draw import HandTracking

app = Flask(__name__)
socketio = SocketIO(app)

# Function to capture video and emit frames to the frontend
def video_feed():
    cap = cv2.VideoCapture(0)
    count = 0
    draw = False
    while True:
        success, image = cap.read()
        hand_tracking = HandTracking(image)

        if not(success):
            break
        hand_tracking_success = hand_tracking.start_drawing()

        draw_current, x, y, frame = hand_tracking_success

        if draw_current:
            count += 1

            if count == 30:
                draw = not(draw)
                count = 0

        if hand_tracking_success is None:
            break

        
        # Convert the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer)


        # Emit the frame to the frontend
        socketio.emit('video_frame', {'image': frame_bytes.decode('utf-8'), 'coordinates': (x, y), 'draw': draw})

    cap.release()

# Route to render the video streaming page
@app.route('/')
def index():
    return render_template('index.html')

# SocketIO event handler for connection
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    socketio.start_background_task(target=video_feed)

if __name__ == '__main__':
    socketio.run(app, debug=True)
