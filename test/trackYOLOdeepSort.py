import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle
import numpy as np

count=0

cy1=600
cy2=800

offset=6

vh_down={}
counter_down=[]


vh_up={}
counter_up=[]

CLASSES = pickle.loads(open("coco_labels.pickle", "rb").read())
#CLASSES = ['bus', 'car', 'motorbike', 'person', 'truck']
CONFIDENCE_THRESHOLD = 0.7
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# initialize the video capture object
video_cap = cv2.VideoCapture('highway.mp4')
# initialize the video writer object

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=25)

colors = []
i = 64
while i > 0:
    colors.append(list(np.random.random(size=3) * 256))
    i = i - 1

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
        cx=int(xmin+xmax)/2
        cy=int(ymin+ymax)/2    

        if cy1<(cy+offset) and cy1 > (cy-offset):
           vh_down[track_id] = track_id  

        if track_id in vh_down:       
            if cy2<(cy+offset) and cy2 > (cy-offset):
                if counter_down.count(id)==0:
                    counter_down.append(track_id)
                
        #####going UP#####     
        if cy2<(cy+offset) and cy2 > (cy-offset):
           vh_up[track_id] = track_id  

        if track_id in vh_up:
           if cy1<(cy+offset) and cy1 > (cy-offset):

                if counter_up.count(track_id)==0:
                    counter_up.append(track_id)      

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colors[int(track_id) % len(colors)], 3)
        cv2.rectangle(frame, (xmin, ymin - 40), (xmax, ymin), colors[int(track_id) % len(colors)], -1)
        #for coco labels
        cv2.putText(frame, f"ID:{track_id} " + CLASSES.get(label[2]) + f" {label[1]:.2f}", (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 4)

        #for custom labels
        #cv2.putText(frame, f"ID:{track_id} " + CLASSES[label[2]] + f" {label[1]:.2f}", (xmin + 5, ymin - 8),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 4)

        numOfObj = numOfObj + 1 
    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    cv2.putText(frame, f"#objects: {numOfObj}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    cv2.line(frame,(100,cy2),(1800,cy2),(255,255,255),3)
    cv2.line(frame,(100,cy1),(1800,cy1),(255,255,255),3)

    cv2.putText(frame,('goingdown:-')+str(len(counter_down)),(60,140),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,255,255),2)
    cv2.putText(frame,('goingup:-')+str(len(counter_up)),(60,180),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,255,255),2)
    # show the frame to our screen
    frame = cv2.resize(frame, (1280, 720))  
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
cv2.destroyAllWindows()