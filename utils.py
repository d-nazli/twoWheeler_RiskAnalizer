import cv2
import numpy as np
from detectionClass import YOLODetection
from DenseOpticalFlow import DenseFlow
import time

prev_time = None
prev_speed = 0.0
gain = 4

fov = 45
focal_length = 1545.10  # px (Önceki hesaplamadan)
object_real_width = 0.75  # metre (Örnek: Motor genişliği)
object_pixel_width = None

yolo_detector = YOLODetection()
dense_flow = DenseFlow()
cap = cv2.VideoCapture("videos/wp1.mp4")
prev_frame = None
draw = True
i = 0

fps = int(cap.get(cv2.CAP_PROP_FPS))  # Saniyedeki kare sayısı

total_x_speed = 0.0
total_y_speed = 0.0
frame_counter = 0
total_speed = []
prev_distance = None
distance = None
speed = 0
metre_list = []


def compute_distance(focal_length_px, object_real_width_m, object_pixel_width):

    if object_pixel_width is None or object_pixel_width <= 0:
        return None

    distance_m = (object_real_width_m * focal_length_px) / object_pixel_width

    if distance_m < 0.5 or distance_m > 100:
        return None

    return round(distance_m, 2)



def compute_pixel_to_meter(focal_length: float, object_distance: float, sensor_width: float, image_width: int) -> float:

    return (object_distance * sensor_width) / (focal_length * image_width)

prev_speed_kmh = 0
def compute_speed(prev_distance, distance, dt, gain=1.0, alpha=0.7):

    global prev_speed

    if prev_distance is None or distance is None or dt is None or dt <= 0:
        return prev_speed

    diff = abs(distance - prev_distance)

    if diff < 0.05 or diff > 1.5:
        diff = 0.0

    instant_speed = (diff / dt) * 3.6 * gain
    print(instant_speed,"instant speed")
    speed_kmh = alpha * prev_speed + (1 - alpha) * instant_speed
    prev_speed = speed_kmh
    print(speed_kmh," speed KMH")


    return round(speed_kmh, 2)



def convert_bayesian_module(distance, director, speed):
    metric_motorcycle = {}

    if distance > 12:
        metric_motorcycle["motor_uzaklık"] = "uzak"
    elif 8 < distance <= 12:
        metric_motorcycle["motor_uzaklık"] = "orta"
    else:
        metric_motorcycle["motor_uzaklık"] = "yakın"

    if director:
        metric_motorcycle["motor_yön"] = director.strip().lower()
    else:
        metric_motorcycle["motor_yön"] = "ortada"

    metric_motorcycle["motor_yönelme"] = "sabit"

    if speed > 50:
        metric_motorcycle["motor_hızı"] = "yüksek"
    elif 20 < speed <= 50:
        metric_motorcycle["motor_hızı"] = "sabit"
    else:
        metric_motorcycle["motor_hızı"] = "düşük"

    return metric_motorcycle

