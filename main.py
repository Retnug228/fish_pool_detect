from ultralytics import YOLO
import cv2
import yaml
import numpy as np

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CAMERA_URL = config["camera_url"]
ZONE_POINTS = np.array(config["zone_points"], np.int32)
CONFIDENCE = config["confidence"]

model = YOLO("yolo_model/yolo11m.pt")

cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    raise RuntimeError("Не удалось подключиться к камере")


def point_in_zone(point, zone):
    return cv2.pointPolygonTest(zone, point, False) >= 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    alert = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls] == "person" and conf > CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                color = (0, 255, 0)

                if point_in_zone(center, ZONE_POINTS):
                    color = (0, 0, 255)
                    alert = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, center, 5, color, -1)

    cv2.polylines(frame, [ZONE_POINTS], True, (255, 255, 0), 2)

    if alert:
        cv2.putText(frame, "Человек у бассейна", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Fish Pool Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
