import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle

#CLASSES = pickle.loads(open("coco_labels.pickle", "rb").read())
CLASSES = ['bus', 'car', 'motorbike', 'person', 'truck']
CONFIDENCE_THRESHOLD = 0.3
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# initialize the video capture object
video_cap = cv2.VideoCapture('/home/zk/Downloads/car2.mp4')
# initialize the video writer object

# load the pre-trained YOLOv8n model
model = YOLO("yoloCar.pt")
tracker = DeepSort(max_age=25)

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    # run the YOLO model on the frame
    detections = model(frame)[0]

    # initialize the list of bounding boxes and confidences
    results = []

    ######################################
    # DETECTION
    ######################################

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        # add the bounding box (x, y, w, h), confidence and class id to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    ######################################
    # TRACKING
    ######################################

    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)
    # loop over the tracks
    numOfObj = 0
    for track,label in zip(tracks, results):
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 80, ymin), GREEN, -1)
        cv2.putText(frame, f"ID:{track_id}-", (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        
        #for coco labels
        #cv2.putText(frame, CLASSES.get(label[2]), (xmin + 30, ymin - 8),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        #for custom labels
        cv2.putText(frame, CLASSES[label[2]], (xmin + 50, ymin - 8),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        cv2.putText(frame, f"conf: {label[1]:.2f}", (xmin + 90, ymin - 8),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        numOfObj = numOfObj + 1 
    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    
    cv2.putText(frame, f"Total: {numOfObj}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # show the frame to our screen
    frame = cv2.resize(frame, (1280, 720))  
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
cv2.destroyAllWindows()