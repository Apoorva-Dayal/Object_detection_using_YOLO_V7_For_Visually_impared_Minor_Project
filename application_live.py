# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2 as cv
import numpy as np
from wandb import Classes
from gtts import gTTS
import os

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save('output.mp3')
    os.system('mpg123 output.mp3')
    # os.system('mpg123')

net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

class DetectionApp(App):
    
    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.start_button = Button(text='Start')
        self.start_button.bind(on_press=self.start_detection)
        self.result_label = Label()

        layout.add_widget(self.image)
        layout.add_widget(self.start_button)
        layout.add_widget(self.result_label)

        self.capture = cv.VideoCapture(0)
        self.is_detecting = False  # Initialize is_detecting to False

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update at 30 FPS

        return layout

    def start_detection(self, instance):
        self.is_detecting = True

    def update(self, dt):
        if self.is_detecting:
            ret, frame = self.capture.read()
            frame = cv.resize(frame, None, fx=1.5, fy=1.5)
            height, width, channel = frame.shape

            blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

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
                    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv.putText(frame, label, (x, y + 30), font, 3, color, 3)

            buf1 = cv.flip(frame, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    DetectionApp().run()
