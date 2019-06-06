import cv2 as cv
import numpy as np
import os
import argparse

#finish adding argparse
#finish changin argument names back
#finish testing output videos and images


parser = argparse.ArgumentParser(description='Inference on an image/video/webcam')

parser.add_argument('--image',   help='Image. Specifiy image location')
parser.add_argument('--video',   help='Video. Specifiy video location')
args = parser.parse_args()

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# Load names of classes
classesFile = "original_yolo/coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
 
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "original_yolo/yolov3.cfg";
modelWeights = "original_yolo/yolov3.weights";
 
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

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


outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)
    
# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, 
                                cv.VideoWriter_fourcc('M','J','P','G'), 
                                30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


## some opencv stuff

winName = 'DL OD with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 1000, 1000)


while cv.waitKey(1) < 0:
     
    # get frame from the video
    hasFrame, frame = cap.read()
     
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break
 
    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
 
    # Sets the input to the network
    net.setInput(blob)
 
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
 
    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)
 
    # Put efficiency information. The function getPerfProfile returns the 
    # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
 
    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))

