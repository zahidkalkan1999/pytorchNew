import cv2
from ultralytics import YOLO
import datetime

# Load the YOLOv8 model
model = YOLO('yoloCar.pt')

# Open the video file
cap = cv2.VideoCapture('/home/pc/Downloads/highway.mp4')

# Loop through the video frames
while cap.isOpened():
    start = datetime.datetime.now()
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        #frame = cv2.resize(frame, (1920, 1080))  
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.7, iou=0.5, tracker="bytetrack.yaml")
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        cv2.putText(annotated_frame, f"#objects: {len(results[0])}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Display the annotated frame

        # end time to compute the fps
        end = datetime.datetime.now()
        # show the time it took to process 1 frame
        # calculate the frame per second and draw it on the frame
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(annotated_frame, fps, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    
        annotated_frame = cv2.resize(annotated_frame, (1280, 720))  
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()