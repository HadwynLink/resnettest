import cv2
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
from torchvision.transforms import v2 
import torchvision

def imshow(image):
    #image = image / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(image, (1, 0, 2)))
    plt.show()

net = cv2.dnn.readNetFromTorch("modelSave.pth")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

img = cv2.imread("data/SPARK/train+val/img077470.jpg")

#imshow(img)


blob = cv2.dnn.blobFromImage(img)

#imshow(img)

net.setInput(blob)

output = net.forward()

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(output, mask=None)
print(output)
print(maxLoc)