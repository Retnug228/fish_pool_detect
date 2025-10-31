import time
import threading
import queue
from datetime import datetime
from ultralytics import YOLO
import cv2
import yaml
import numpy as np
import csv
import os


# ---------- Конфигурация ----------
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- CSV ----------
def init_csv(filename="csv/people_log_with_zones.csv"):
    if not os.path.exists(filename):
        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "event", "zones", "time", "duration"])
    return filename


def log_event(pid, event, zones=None, duration=None, filename="people_log_with_zones.csv"):
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            pid,
            event,
            ",".join(zones) if zones else "",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{duration:.2f}s" if duration else ""
        ])


# ---------- Зоны ----------
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


# ---------- Детекция ----------
def get_person_detections(result, model, confidence):
    people = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if hasattr(box, "id") and box.id is not None else None

        if model.names[cls] == "person" and conf > confidence and track_id is not None:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2)//2, (y1 + y2)//2)
            people.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "conf": conf
            })
    return people


def analyze_zones(person, zones):
    zones_inside = []
    for z in zones:
        if point_in_zone(person["center"], z["points"]):
            zones_inside.append(z["name"])
    return zones_inside


# ---------- Логика слежения ----------
def update_tracked_people(people, tracked_people, csv_file):
    current_ids = {p["track_id"] for p in people}

    # Новые люди
    for p in people:
        pid = p["track_id"]
        zones_inside = analyze_zones(p, zones)
        if pid not in tracked_people:
            tracked_people[pid] = {
                "start_time": datetime.now(),
                "zones": zones_inside,
                "last_seen": datetime.now()
            }
            print(f"[ARRIVAL] Человек {pid} вошёл в {zones_inside}")
            log_event(pid, "arrival", zones_inside, filename=csv_file)
        else:
            tracked_people[pid]["last_seen"] = datetime.now()
            tracked_people[pid]["zones"] = zones_inside

    # Проверяем, кто ушёл
    gone_ids = []
    now = datetime.now()
    for pid, info in tracked_people.items():
        if pid not in current_ids:
            duration = (now - info["start_time"]).total_seconds()
            print(f"[DEPARTURE] Человек {pid} покинул все зоны. Был {duration:.2f} сек.")
            log_event(pid, "departure", info["zones"], duration, csv_file)
            gone_ids.append(pid)

    for gid in gone_ids:
        tracked_people.pop(gid)


# ---------- Отрисовка ----------
def draw_detections(frame, people, zones):
    alerts = []
    for p in people:
        zones_inside = analyze_zones(p, zones)
        color = (0, 255, 0)
        if zones_inside:
            alerts.extend(zones_inside)
            for z in zones:
                if z["name"] in zones_inside:
                    color = z["color"]
                    break

        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, p["center"], 5, color, -1)
        cv2.putText(frame, f"ID {p['track_id']}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return alerts


# ---------- Обработка кадра ----------
def process_frame(result, model, zones, confidence, tracked_people, csv_file):
    frame = result.orig_img.copy()
    people = get_person_detections(result, model, confidence)
    update_tracked_people(people, tracked_people, csv_file)
    alerts = draw_detections(frame, people, zones)
    return frame, alerts


# ---------- Потоки ----------
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


def frame_processor(frame_queue, model, zones, confidence, stop_event, csv_file):
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

        frame, alert = process_frame(results[0], model, zones, confidence, tracked_people, csv_file)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        draw_zones(frame, zones)

        if alert:
            cv2.putText(frame, f"Человек в: {', '.join(set(alert))}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Fish Pool Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()


# ---------- Основной запуск ----------
def main():
    config = load_config()
    global zones
    zones = prepare_zones(config.get("zones", []))
    confidence = config.get("confidence", 0.5)
    camera_url = config.get("camera_url")

    model = YOLO(config.get("yolo_model", "yolo_model/yolo11s.pt"))
    csv_file = init_csv()

    frame_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    reader_thread = threading.Thread(target=frame_reader, args=(camera_url, frame_queue, stop_event))
    processor_thread = threading.Thread(target=frame_processor, args=(frame_queue, model, zones, confidence, stop_event, csv_file))

    reader_thread.start()
    processor_thread.start()

    reader_thread.join()
    processor_thread.join()


if __name__ == "__main__":
    main()
