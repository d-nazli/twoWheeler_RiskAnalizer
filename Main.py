import cv2
import numpy as np
from ultralytics import YOLO
from detectionClass import YOLODetection
from DenseOpticalFlow import DenseFlow
import math
import time
from utils import *
from risk_analyzer import ALLANALIZER
from risk_display import risk_kutusu_ekle
from collections import deque

pixel_width_history = deque(maxlen=5)
prev_speed_kmh = list()
risk = None
director = None
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cur_time = time.time()  # sistem zamanı (gerçek zaman farkı için)
    if prev_time is not None:
        frame_dt = np.clip(cur_time - prev_time, 1/40, 1/20)

        if frame_dt > 0:
            real_fps = 1 / frame_dt
            print(f"Gerçek FPS: {real_fps:.2f}")
    else:
        frame_dt = 1.0 / fps  # ilk kare için tahmini zaman farkı

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (1300, 1920), 255, -1)

    detections = yolo_detector.detect_objects(frame)  # isteğe bağlı maske yollanabilir.
    frame_counter += 1
    print(frame_counter)
    if detections and prev_frame is not None and i > 20:
        x1, y1, x2, y2 = detections[0]
        cropped_frame = frame[y1:y2, x1:x2]
        HalfW = abs(x2+x1)/2
        if HalfW < 640:
            director = "sol"
        else:
            director = "sağ"

        current_width = x2 - x1

        if len(pixel_width_history) > 0:
            prev_mean = sum(pixel_width_history) / len(pixel_width_history)
            if prev_mean > 0 and abs(current_width - prev_mean) / prev_mean > 0.3:
                pass
            else:
                pixel_width_history.append(current_width)
        else:
            pixel_width_history.append(current_width)

        object_pixel_width = sum(pixel_width_history) / len(pixel_width_history)
        print(f"[FRAME {frame_counter}] object_pixel_width={object_pixel_width:.2f}")

        prev_image, cropped_frame = dense_flow.size_matching(prev_frame, cropped_frame)

        flow = dense_flow.compute_dens_flow(prev_image, cropped_frame)


        mask_flow = flow[y1:y2, x1:x2]
        if mask_flow.size == 0:
            x_speed, y_speed = 0.0, 0.0
        else:
            mean_flow = np.mean(mask_flow, axis=(0, 1))  # [mean_x, mean_y]
            x_speed = float(mean_flow[0])
            y_speed = float(mean_flow[1])

        total_x_speed += abs(x_speed)
        total_y_speed += abs(y_speed)

        new_distance = compute_distance(focal_length, object_real_width, object_pixel_width)
        distance = new_distance  # bu frame'in uzaklığı

        if prev_distance is not None and new_distance is not None and prev_time is not None:
            dt = cur_time - prev_time
            speed = compute_speed(prev_distance, new_distance, dt, gain=gain)
            speed = speed/3
            prev_speed_kmh.append(speed)



        print(f"[FRAME {frame_counter}] new_distance={new_distance}, prev_distance={prev_distance}, speed={speed:.2f}")


        if distance and prev_distance:
            cv2.rectangle(frame, (10, 10), (330, 90), (0, 0, 0), -1)  # siyah arka plan
            cv2.putText(frame, f"UZAKLIK: {distance:.2f} m", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"HIZ: {speed:.2f} km/h", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            converted_risk = convert_bayesian_module(distance, director, speed)
            risk = ALLANALIZER(converted_risk)
            risk_kutusu_ekle(frame, risk)



        optical_flow_colormap = dense_flow.vis_optical_flow_colormap(prev_image)
        optical_flow_colormap = cv2.resize(optical_flow_colormap, dsize=(480, 480))
        cv2.imshow("Optical Flow", optical_flow_colormap)

    if detections:
        x1, y1, x2, y2 = detections[0]
        prev_frame = frame[y1:y2, x1:x2]  # Yeni prev_frame'i güncelle

    else:
        print("No detections - Keeping prev_frame unchanged")  # Debugging için

    i += 1

    if detections and draw:
        frame = yolo_detector.draw_objects(frame, detections)

    show_frame = frame.copy()

    print(
        f"[FRAME {frame_counter}] object_pixel_width={object_pixel_width if 'object_pixel_width' in locals() else 'N/A'}, distance={distance if 'distance' in locals() else 'N/A'}, prev_distance={prev_distance if 'prev_distance' in locals() else 'N/A'}, speed={speed if 'speed' in locals() else 'N/A'}")

    if 'new_distance' in locals() and new_distance is not None:
        prev_distance = new_distance

    cv2.imshow("Original Frame", show_frame)
    prev_time = cur_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(f"object_pixel_width={object_pixel_width}")
print(f"distance={distance}")
print(f"prev_distance={prev_distance}")
print(f"speed={speed}")

cv2.destroyAllWindows()


print(prev_speed_kmh)
print("Min: ",min(prev_speed_kmh))
print("Max: ",max(prev_speed_kmh))
