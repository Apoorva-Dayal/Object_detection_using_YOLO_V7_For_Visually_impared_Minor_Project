import cv2 as cv
import numpy as np
from wandb import Classes
from gtts import gTTS
import os

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save('output.mp3')
    os.system('mpg123 output.mp3')

net = cv.dnn.readNet("yolov3.weights","yolov3.cfg")
clasees = []
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_name = net.getLayerNames()
output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

img = cv.imread("sample_2.jpg") #change image according to the requirement
img = cv.resize(img, None, fx=.3, fy=.4)
height, width, channel = img.shape

blob = cv.dnn.blobFromImage(
    img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob) 
outs = net.forward(output_layer)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv.FONT_HERSHEY_PLAIN
class_counts = {label: 0 for label in classes}

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        class_counts[label] += 1
        text_to_speech(f"Detected {label}")
        color = colors[class_ids[i]]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, label, (x, y + 30), font, 3, color, 3)

for label, count in class_counts.items():
    if count > 0:
        print(f"Number of {label}s detected: {count}")
        text_to_speech(f"Number of {label}s detected: {count}")
  
cv.imshow("IMG", img)
cv.waitKey(10000000)
