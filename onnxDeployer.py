import onnxruntime as rt
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import os

correct = 0
total = 0


#define the priority order for the execution providers
# prefer CUDA Execution Provider over CPU Execution Provider
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# initialize the model.onnx
sess = rt.InferenceSession("lasttest.onnx", providers=EP_list)

# get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
output_name = sess.get_outputs()[0].name

# get the inputs metadata as a list of :class:`onnxruntime.NodeArg`
input_name = sess.get_inputs()[0].name


def imshow(image):
    #image = image / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(image, (1, 0, 2)))
    plt.show()


with open('data/SPARK/newtrain.csv', newline='') as csvfile:

    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
        lename = row[0]
        if (lename != "filename"):
            image_data = cv2.imread("data/SPARK/train+val/" + lename)
            img = np.float32(image_data)
            img = img.reshape(1,3,256,256)
            #fun = x
            #fun = torchvision.utils.make_grid(fun.cpu())
            #fun = fun[:, :, :3]
            #imshow(fun)
            # inference run using image_data as the input to the model
            detections = sess.run([output_name], {input_name: img})[0]
            index = 0
            largestIndex = 0
            largestNum = detections[0][0]
            for i in detections[0]:
                if i > largestNum:
                    largestIndex = index
                    largestNum = i
                index += 1
            if largestIndex == int(row[1]):
                correct += 1
            total += 1
            print("Correct: " , correct, "Out of ", total, ". Expected: ", largestIndex, "Actual: ", row[1])

print("Percent correct: ", (correct/total))