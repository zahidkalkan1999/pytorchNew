from ultralytics import YOLO

# Load a model
model = YOLO('yolov8l.pt')  # load a pretrained model

# Train the model
results = model.train(data='/home/zk/Downloads/yolov8/data.yaml', epochs=80, imgsz=640, batch=2, optimizer='Adam', lr0=0.001)