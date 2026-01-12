import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetection:
    def __init__(self, model_path=r"C:\2209Duygu (2)\2209Duygu\new_best.pt", conf_threshold=0.4):
        self.model = YOLO(model_path,verbose=False)
        self.conf_threshold = conf_threshold
    
    def detect_objects(self, frame, mask=None):

        if mask is not None:
            input_image = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            input_image = frame

        results = self.model(input_image)
        detected_objects = []
        
        for result in results:
            print(("Detection Yapıldı"))
            for box in result.boxes:

                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                print(fr"{cls_id} Tespit edildi")
                
                if cls_id == 1 and conf > self.conf_threshold:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    detected_objects.append((x1, y1, x2, y2))
                    
        return detected_objects
    
    def draw_objects(self, frame, detections):

        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return frame

