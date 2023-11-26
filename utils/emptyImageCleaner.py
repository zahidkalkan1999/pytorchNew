from bs4 import BeautifulSoup
from imutils import paths
import os 

# define the base path to the *original* input dataset and then use
# the base path to derive the image and annotations directories
ORIG_BASE_PATH = "/home/zk/data"

ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "train"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "train"])

# grab all image paths in the input images directory
imagePaths = list(paths.list_images(ORIG_IMAGES))
print(imagePaths)

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# show a progress report
	print(f"[INFO] processing image {i + 1}/{len(imagePaths)}...")
	# extract the filename from the file path and use it to derive
	# the path to the XML annotation file
	filename = imagePath.split(os.path.sep)[-1]
	filename = filename[:filename.rfind(".")]
	annotPath = os.path.sep.join([ORIG_ANNOTS, f"{filename}.xml"])
	# load the annotation file, build the soup, and initialize our
	# list of ground-truth bounding boxes
	contents = open(annotPath).read()
	soup = BeautifulSoup(contents, "html.parser")
	
    # loop over all 'object' elements
	if len(soup.find_all("object")) == 0:
		os.remove(imagePath)
		os.remove(annotPath)