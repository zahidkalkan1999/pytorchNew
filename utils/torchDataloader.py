import config
from imutils import paths
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import shutil
import os

def visualizeBatch(batch, classes, dataset_type):
	# initialize a figure
	fig = plt.figure(f"{dataset_type} batch",
		figsize=(config.BATCH_SIZE, config.BATCH_SIZE))
	# loop over the batch size
	for i in range(0, config.BATCH_SIZE):
		# create a subplot
		ax = plt.subplot(2, 4, i + 1)
		# grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
		image = batch[0][i].cpu().numpy()
		image = image.transpose((1, 2, 0))
		image = (image * 255.0).astype("uint8")
		# grab the label id and get the label from the classes list
		idx = batch[1][i]
		label = classes[idx]
		# show the image along with the label
		plt.imshow(image)
		plt.title(label)
		plt.axis("off")
	# show the plot
	plt.tight_layout()
	plt.show()

def dataLoader(dataPath):
        # initialize our data augmentation functions
        resize = transforms.Resize(size=config.INPUT_DIMS)
        hFlip = transforms.RandomHorizontalFlip(p=0.25)
        vFlip = transforms.RandomVerticalFlip(p=0.25)
        rotate = transforms.RandomRotation(degrees=15)

        # initialize our training and validation set data augmentation
        # pipeline
        trainTransforms = transforms.Compose([resize, hFlip, vFlip, rotate,
                transforms.ToTensor()])

        # initialize the training and validation dataset
        print("[INFO] loading the training and validation dataset...")
        trainDataset = ImageFolder(dataPath, transform=trainTransforms)

        print(f"[INFO] training dataset contains {len(trainDataset)} samples...")

        # calculate the train/validation split
        print("[INFO] generating the train/validation split...")
        numValSamples = int(len(trainDataset) * config.VAL_SPLIT)
        numTrainSamples = int(len(trainDataset) - numValSamples)

        (trainData, valData) = random_split(trainDataset,
                [numTrainSamples, numValSamples],
                generator=torch.Generator().manual_seed(42))

        # create training and validation set dataloaders
        print("[INFO] creating training and validation set dataloaders...")
        trainDataLoader = DataLoader(trainData, batch_size=config.BATCH_SIZE, shuffle=True)
        valDataLoader = DataLoader(valData, batch_size=config.BATCH_SIZE)
		
        # grab a batch from both training and validation dataloader
        #trainBatch = next(iter(trainDataLoader))
        #valBatch = next(iter(valDataLoader))
		
        # visualize the training and validation set batches
        print("[INFO] visualizing training and validation batch...")
        #visualizeBatch(trainBatch, trainDataset.classes, "train")
        #visualizeBatch(valBatch, trainDataset.classes, "val")
		
        return trainDataLoader, valDataLoader

