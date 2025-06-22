import os
import json
import base64
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import yaml

def load_labels():
    with open("./function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)
    labels_spec = functionconfig['metadata']['annotations']['spec']
    
    loaded_labels = json.loads(labels_spec)
    labels_map = {}
    label_types_map = {}
    for item in loaded_labels:
        labels_map[item['id']] = item['name']
        label_types_map[item['id']] = item['type'] # 儲存 type
    return labels_map, label_types_map


# Load model
MODEL_PATH = "./Yolov11.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
labels, label_types = load_labels() # 接收兩個返回值

def init_context(context):
    context.logger.info("Initializing YOLOv11 model...")
    context.user_data.model = model
    context.user_data.labels = labels
    context.user_data.label_types = label_types
    context.logger.info("Model loaded.")

def handler(context, event):
    context.logger.info("Handling inference request...")

    # Parse request data
    data = event.body
    if isinstance(data, bytes):
        data = json.loads(data.decode())

    if "image" not in data:
        return context.Response(
            body=json.dumps({"error": "Missing 'image' field in request."}),
            headers={"Content-Type": "application/json"},
            status_code=400
        )

    try:
        image_bytes = base64.b64decode(data["image"])
        threshold = float(data.get("threshold", 0.5))
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return context.Response(
            body=json.dumps({"error": f"Failed to process image: {str(e)}"}),
            headers={"Content-Type": "application/json"},
            status_code=400
        )

    # Run inference
    results = context.user_data.model(image)[0]

    detections = []
    for box in results.boxes:
        # 這裡的 box.xyxy[0].tolist() 已經是 [x1, y1, x2, y2] 的列表了
        # 您可以直接使用它作為 "points" 的值
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        if conf >= threshold:
            detections.append({
                "confidence": str(conf), # 將浮點數轉換為字串
                "label": context.user_data.labels.get(cls_id, f"class_{cls_id}"),
                "points": box.xyxy[0].tolist(), # 直接使用 box.xyxy[0].tolist()
                "type": context.user_data.label_types.get(cls_id, "rectangle")
            })

    return context.Response(
        body=json.dumps(detections),
        headers={"Content-Type": "application/json"},
        status_code=200
    )
