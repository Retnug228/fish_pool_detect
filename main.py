import time
import threading
import queue
from datetime import datetime
from ultralytics import YOLO
import cv2
import yaml
import numpy as np


#   Конфигурация  
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


#   Зоны  
def prepare_zones(zone_list):
    zones = []
    for z in zone_list:
        zones.append({
            "name": z["name"],
            "color": tuple(z.get("color", [0, 0, 255])),
            "points": np.array(z["points"], np.int32)
        })
    return zones


def point_in_zone(point, zone_points):
    return cv2.pointPolygonTest(zone_points, point, False) >= 0


def draw_zones(frame, zones):
    for z in zones:
        cv2.polylines(frame, [z["points"]], True, z["color"], 2)


def draw_person(frame, box, center, color):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, center, 5, color, -1)


#   Обработка  
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

    gone_ids = set(tracked_people.keys()) - current_ids
    for gid in gone_ids:
        departure_time = datetime.now()
        print(f"[DEPARTURE] Человек {gid} покинул зону в {departure_time.strftime('%Y-%m-%d %H:%M:%S')}")
        tracked_people.pop(gid)

    return frame, alert


#   Потоки  
def frame_reader(camera_url, frame_queue, stop_event):
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        raise RuntimeError("Не удалось подключиться к камере")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()


def frame_processor(frame_queue, model, zones, confidence, stop_event):
    prev_time = time.time()
    tracked_people = {}

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()

        results = model.track(frame, persist=True)
        if not results:
            continue

        frame, alert = process_frame(results[0], zones, confidence, tracked_people)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        draw_zones(frame, zones)

        if alert:
            cv2.putText(frame, f"Человек в зоне: {', '.join(alert)}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Fish Pool Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()


#   Основной запуск  
def main():
    config = load_config()
    zones = prepare_zones(config.get("zones", []))
    confidence = config.get("confidence", 0.5)

    global model
    model = YOLO(config.get("yolo_model", "yolo_model/yolo11s.pt"))

    frame_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    reader_thread = threading.Thread(target=frame_reader, args=(config["camera_url"], frame_queue, stop_event))
    processor_thread = threading.Thread(target=frame_processor, args=(frame_queue, model, zones, confidence, stop_event))

    reader_thread.start()
    processor_thread.start()

    reader_thread.join()
    processor_thread.join()


if __name__ == "__main__":
    main()
