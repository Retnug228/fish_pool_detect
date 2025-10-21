import time
from ultralytics import YOLO
import cv2
import yaml
import numpy as np

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CAMERA_URL = config["camera_url"]
CONFIDENCE = config["confidence"]
ZONE_POINTS_LIST = config.get("zones", [])

# Подготовка зон
zones = []
for z in ZONE_POINTS_LIST:
    zones.append({
        "name": z["name"],
        "color": tuple(z.get("color", [0, 0, 255])),
        "points": np.array(z["points"], np.int32)
    })

# Функция проверки попадания в зону
def point_in_zone(point, zone_points):
    return cv2.pointPolygonTest(zone_points, point, False) >= 0

# Модель
model = YOLO(config.get("yolo_model", "yolo_model/yolo11s.pt"))

# Поток
cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    raise RuntimeError("Не удалось подключиться к камере")

prev_time = time.time()

# Главный цикл
for result in model.track(source=CAMERA_URL, stream=True):
    frame = result.orig_img.copy()
    alert = []

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if model.names[cls] == "person" and conf > CONFIDENCE:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2)//2, (y1 + y2)//2)
            color = (0, 255, 0)

            for z in zones:
                if point_in_zone(center, z["points"]):
                    color = z["color"]
                    alert.append(z["name"])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, center, 5, color, -1)

            if alert:
                print(f"[ALERT] Человек в зоне(ах): {', '.join(alert)}")

    for z in zones:
        cv2.polylines(frame, [z["points"]], True, z["color"], 2)

    if alert:
        cv2.putText(frame, f"Человек в зоне: {', '.join(alert)}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Fish Pool Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
