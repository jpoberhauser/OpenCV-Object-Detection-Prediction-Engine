#od.py
#https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/

# can we make s cript that shows darknet predictions on single iamges, webcam, and video?

import cv2 as cv
import numpy as np


confThreshold = .25
#non maximum supression 
nmsThreshold = 0.40
inpWidth = 416
inpHeight = 416

classesFile = "coco.names"
classes = None

# read classes file (80 classes on coco)
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def getOutputsNames(net):
    layerNames = net.getLayerNames()
    return [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right,bottom):
    cv.rectangle(frame, (left, top), (right, bottom),(255,178,50),3)

    label = "%.2f" % conf

    if classes:
        assert(classId<len(classes))
        label = "%s:%s"%(classes[classId],label)

    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),3)



def postprocess(frame, outs):
    #yolo outputs centerX, centerY, 
    # width, height, conf, class1, class2...class80
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            #gets the ids of classes
            classID = np.argmax(scores)
            #probability of classID
            confidence = scores[classID]


            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                left = int(centerX - width/2)
                top = int(centerY - height /2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indeces = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indeces:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)




modelConf = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


## some opencv stuff

winName = 'DL OD with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 1000, 1000)


#use the wecam as the spurce of videp
cap = cv.VideoCapture(0)


while cv.waitKey(1) < 0:
    #capture frame-by-frame
    hasFrame, frame = cap.read()
    #swapRB parameter, no cropping just resizing
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight),[0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)
    cv.imshow(winName, frame)