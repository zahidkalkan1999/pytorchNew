import datetime
from ultralytics import YOLO
import cv2
import pickle

#for coco labels
#CLASSES = pickle.loads(open("coco_labels.pickle", "rb").read())

#for custom model
CLASSES = ['bus', 'car', 'motorbike', 'person', 'truck']

CONFIDENCE_THRESHOLD = 0.9
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# initialize the video capture object
video_cap = cv2.VideoCapture('/home/pc/Downloads/highway.mp4')
# initialize the video writer object

# load the pre-trained YOLOv8n model
model = YOLO("yoloCar.pt")

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

    ######################################
    # DETECTION
    ######################################

    # loop over the detections
    numOfObj = 0
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) == 1:
            continue

        # if the confidence is greater than the minimum confidence,
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
    
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 3)
        cv2.rectangle(frame, (xmin, ymin - 40), (xmax, ymin), GREEN, -1)
        
        #for coco labels
        #cv2.putText(frame, f"{CLASSES.get(class_id)}", (xmin + 5, ymin - 8),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        #for custom model
        
        cv2.putText(frame, f"{CLASSES[class_id]}", (xmin + 5, ymin - 8),
		cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 4)
        
        cv2.putText(frame, f"%{100*confidence:.2f}", (xmin + 80, ymin - 8),
		cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 4)
        numOfObj = numOfObj + 1 
    # end time to compute the fps
    end = datetime.datetime.now()
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    
    cv2.putText(frame, f"Total: {numOfObj}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
cv2.destroyAllWindows()