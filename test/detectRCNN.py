# import the necessary packages
import sys
sys.path.append("/home/zk/pytorch/pytorchlast/")
from utils.RCNNmodel import create_model

import datetime
from torchvision.models import detection
import numpy as np
import argparse
import torch
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="frcnn-mobilenet",
	choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
	help="name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="coco_labels.pickle",
	help="path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
#CLASSES = pickle.loads(open(args["labels"], "rb").read())

CLASSES = [
'bus', 'car', 'motorbike', 'person', 'truck'
]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# initialize a dictionary containing model name and its corresponding 
# torchvision function call
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn
}
# load a pretrained model and set it to evaluation mode
#model = MODELS[args["model"]](pretrained=True, progress=True,
	#pretrained_backbone=True).to(DEVICE)

# load the model and the trained weights for custom dataset
model = create_model(num_classes=5).to(DEVICE)
checkpoint = torch.load('rcnnCar.pth')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

video_cap = cv2.VideoCapture('/home/zk/Downloads/car5.mp4')

while True:
	start = datetime.datetime.now()

	ret, frame = video_cap.read()

	if not ret:
		break

	# load the image from disk
	orig = frame.copy()
	# convert the image from BGR to RGB channel ordering and change the
	# image from channels last to channels first ordering
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = frame.transpose((2, 0, 1))
	# add the batch dimension, scale the raw pixel intensities to the
	# range [0, 1], and convert the image to a floating point tensor
	frame = np.expand_dims(frame, axis=0)
	frame = frame / 255.0
	frame = torch.FloatTensor(frame)
	# send the input to the device and pass the it through the network to
	# get the detections and predictions
	frame = frame.to(DEVICE)
	detections = model(frame)[0]
	num = 0
	# loop over the detections
	for i in range(0, len(detections["boxes"])):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections["scores"][i]
		
		if args["confidence"] > confidence:
			continue
		
		box = detections["boxes"][i].detach().cpu().numpy()
		(xmin, ymin, xmax, ymax) = box.astype("int")
		class_id = int(detections["labels"][i])

		cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), GREEN, 2)
		cv2.rectangle(orig, (xmin, ymin - 20), (xmin + 80, ymin), GREEN, -1)
		cv2.putText(orig, CLASSES[class_id], (xmin + 5, ymin - 8),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

		cv2.putText(orig, f"%{100*confidence:.2f}", (xmin + 120, ymin - 8),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

		num = num + 1
	cv2.putText(orig, str(num), (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	# end time to compute the fps
	end = datetime.datetime.now()
	# calculate the frame per second and draw it on the frame
	fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
	cv2.putText(orig, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

	# show the frame to our screen
	orig = cv2.resize(orig, (1280, 720))  
	cv2.imshow("orig", orig)
	if cv2.waitKey(1) == ord("q"):
		break

video_cap.release()
cv2.destroyAllWindows()
