import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import csv
import os

# ======= Настройки =======
ZONE_POINTS = np.array([[0, 0], [0, 1080], [1920, 1080], [1920, 0]])
CONFIDENCE = 0.5
LOG_FILE = "../csv/zone_log.csv"

model = YOLO("../yolo_model/yolo11s.pt")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "время_прихода", "время_ухода", "продолжительность_сек"])

def is_inside_zone(x, y, zone_points):
    return cv2.pointPolygonTest(zone_points, (x, y), False) >= 0


def log_to_csv(entry_time, exit_time, duration):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)

    person_id = line_count
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([person_id, entry_time, exit_time, duration])


cap = cv2.VideoCapture(0)
person_in_zone = False
entry_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    cv2.polylines(frame, [ZONE_POINTS], isClosed=True, color=(0, 255, 255), thickness=2)

    person_detected_in_zone = False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls] == "person" and conf > CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if is_inside_zone(cx, cy, ZONE_POINTS):
                    person_detected_in_zone = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    if person_detected_in_zone and not person_in_zone:
        person_in_zone = True
        entry_time = datetime.now()
        print(f"[{entry_time.strftime('%H:%M:%S')}] Человек ВОШЕЛ в зону.")

    elif not person_detected_in_zone and person_in_zone:
        person_in_zone = False
        exit_time = datetime.now()
        duration = (exit_time - entry_time).seconds if entry_time else 0
        print(f"[{exit_time.strftime('%H:%M:%S')}] Человек ВЫШЕЛ из зоны. Был в зоне {duration} сек.")
        log_to_csv(entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                   exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                   duration)
        entry_time = None

    cv2.imshow("YOLO Zone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Работа завершена. Данные сохранены в:", LOG_FILE)
