import time
from datetime import datetime
from ultralytics import YOLO
import cv2
import yaml
import numpy as np

# Загрузка конфигурации
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Подготовка зон
def prepare_zones(zone_list):
    zones = []
    for z in zone_list:
        zones.append({
            "name": z["name"],
            "color": tuple(z.get("color", [0, 0, 255])),
            "points": np.array(z["points"], np.int32)
        })
    return zones

# Проверка попадания в зону
def point_in_zone(point, zone_points):
    return cv2.pointPolygonTest(zone_points, point, False) >= 0

# Отрисовка зон
def draw_zones(frame, zones):
    for z in zones:
        cv2.polylines(frame, [z["points"]], True, z["color"], 2)

# Отрисовка человека
def draw_person(frame, box, center, color):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, center, 5, color, -1)

# Обработка кадра
def process_frame(result, zones, confidence, tracked_people):
    frame = result.orig_img.copy()
    alert = []
    current_ids = set()

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if hasattr(box, "id") else None

        if cls is not None and model.names[cls] == "person" and conf > confidence and track_id is not None:
            current_ids.add(track_id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2)//2, (y1 + y2)//2)
            color = (0, 255, 0)

            person_zones = []
            for z in zones:
                if point_in_zone(center, z["points"]):
                    color = z["color"]
                    person_zones.append(z["name"])
                    alert.append(z["name"])

            # Логика прихода
            if track_id not in tracked_people:
                tracked_people[track_id] = {
                    "arrival": datetime.now(),
                    "zones": person_zones
                }
                print(f"[ARRIVAL] Человек {track_id} появился в зоне(ах): "
                      f"{', '.join(person_zones)} в {tracked_people[track_id]['arrival'].strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                tracked_people[track_id]["zones"] = person_zones

            draw_person(frame, box, center, color)

    # Логика ухода
    gone_ids = set(tracked_people.keys()) - current_ids
    for gid in gone_ids:
        departure_time = datetime.now()
        print(f"[DEPARTURE] Человек {gid} покинул зону в {departure_time.strftime('%Y-%m-%d %H:%M:%S')}")
        tracked_people.pop(gid)

    return frame, alert

# Основной цикл
def main():
    config = load_config()
    zones = prepare_zones(config.get("zones", []))
    confidence = config.get("confidence", 0.5)

    global model
    model = YOLO(config.get("yolo_model", "yolo_model/yolo11s.pt"))

    cap = cv2.VideoCapture(config["camera_url"])
    if not cap.isOpened():
        raise RuntimeError("Не удалось подключиться к камере")

    prev_time = time.time()
    tracked_people = {}

    for result in model.track(source=config["camera_url"], stream=True):
        frame, alert = process_frame(result, zones, confidence, tracked_people)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Отрисовка зон
        draw_zones(frame, zones)

        # Предупреждение
        if alert:
            cv2.putText(frame, f"Человек в зоне: {', '.join(alert)}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        cv2.imshow("Fish Pool Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
