import cv2
import numpy as np
from flask import Flask, render_template, request
import time


app = Flask(__name__)

with open("./yolov3.txt", "r") as f:
    classes = f.read().splitlines()

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


scale = 0.00392

net = cv2.dnn.readNet("./yolov3-tiny.weights", "./yolo tiny.cfg")

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    print(label)

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 4)


@app.route("/predict", methods = ['POST'])
def predict():
    f = request.files['image'].read()
    npimg = np.fromstring(f, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = get(image)

    return render_template("index.html" , result = result)

 
def get(image):
    
    Width = image.shape[1]
    Height = image.shape[0]

    blob = cv2.dnn.blobFromImage(image, 1/255, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()

    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    print(len(boxes))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    img_name = "./static/Images/Image-" + str(time.time()) + ".jpg"
    print(img_name)
    cv2.imwrite(img_name, image)
    return img_name[9:]




@app.route("/")
def index():
    return render_template("index.html")

if(__name__ == '__main__'):
    app.run(debug = True)