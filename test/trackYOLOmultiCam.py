import cv2
from ultralytics import YOLO
import numpy as np
import datetime

# Load the models
model = YOLO('yoloCar.pt')

# Define the video files for the trackers
video_file1 = '/home/zk/Downloads/highway2.mp4'  # Path to video file, 0 for webcam
video_file2 = '/home/zk/Downloads/highway.mp4'  # Path to video file, 0 for webcam, 1 for external camera

video1 = cv2.VideoCapture(video_file1)  # Read the video file
video2 = cv2.VideoCapture(video_file2)  # Read the video file

while True:
    start = datetime.datetime.now()

    ret1, frame1 = video1.read()  # Read the video frames
    ret2, frame2 = video2.read()  # Read the video frames
    # Exit the loop if no more frames in either video
    if not ret1 or not ret2:
        break
    
    frame = np.concatenate((frame1, frame2), axis=1)
    # Track objects in frames if available
    results = model.track(frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml")
    res_plotted = results[0].plot()

    cv2.putText(res_plotted, f"Total vehicles: {len(results[0])}", (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # end time to compute the fps
    end = datetime.datetime.now()
	# calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(res_plotted, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    res_plotted = cv2.resize(res_plotted, (1280, 540))  
    cv2.imshow(f"frame", res_plotted)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release video sources
video1.release()
video2.release()
# Clean up and close windows
cv2.destroyAllWindows()