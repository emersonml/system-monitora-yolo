from flask import Flask, request, jsonify
import cv2
import numpy as np
import time

app = Flask(__name__)

# Carrega os arquivos de configuração e pesos da YOLOv4
config_path = 'yolov4.cfg'
weights_path = 'yolov4.weights'
names_path = 'coco.names'

# Carrega os nomes das classes
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Configura a rede YOLOv4
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Função para realizar a detecção de objetos
def detect_objects(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())
    end = time.time()

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype('int')

                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detections.append({
                "class": classes[class_ids[i]],
                "confidence": confidences[i],
                "box": [x, y, w, h]
            })

    return detections, end - start

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_path = data.get('image_path')
    if not image_path:
        return jsonify({"error": "No image path provided"}), 400

    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Could not read image"}), 400

    detections, inference_time = detect_objects(image)

    response = {
        "detections": detections,
        "inference_time": inference_time
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# if __name__ == '__main__':
#     app.run(debug=True)



