import time
import threading
import queue
from datetime import datetime
from ultralytics import YOLO
import cv2
import yaml
import csv
import os
import traceback


# ---------- Конфигурация ----------
def load_config(path="config.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[CONFIG ERROR] Ошибка чтения {path}: {e}")
        return {"camera_url": 0, "confidence": 0.5, "yolo_model": "yolo_model/yolo11s.pt"}


# ---------- CSV ----------
def init_csv(filename="csv/people_log.csv"):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if not os.path.exists(filename):
            with open(filename, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "first_arrival", "last_departure", "total_duration"])
        return filename
    except Exception as e:
        print(f"[CSV INIT ERROR] {e}")
        return filename


def log_final_event(pid, info, filename="csv/people_log.csv"):
    try:
        with open(filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                pid,
                info["first_arrival"].strftime("%Y-%m-%d %H:%M:%S"),
                info["last_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                f"{info['total_duration']:.2f}s"
            ])
    except Exception as e:
        print(f"[CSV WRITE ERROR] {e}")


# ---------- Детекция ----------
def get_person_detections(result, model, confidence):
    people = []
    try:
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
    except Exception as e:
        print(f"[DETECTION ERROR] {e}")
    return people


# ---------- Логика слежения ----------
def update_tracked_people(people, tracked_people, csv_file, lost_timeout=3):
    now = datetime.now()
    current_ids = {p["track_id"] for p in people}

    try:
        # Новые или вернувшиеся
        for p in people:
            pid = p["track_id"]
            if pid not in tracked_people:
                tracked_people[pid] = {
                    "first_arrival": now,
                    "last_seen": now,
                    "last_seen_time": time.time(),
                    "total_duration": 0.0,
                    "in_frame": True
                }
                print(f"[ARRIVAL] Человек {pid} вошёл в кадр ({now.strftime('%H:%M:%S')})")
            else:
                info = tracked_people[pid]
                info["last_seen"] = now
                info["last_seen_time"] = time.time()
                if not info["in_frame"]:
                    print(f"[RETURN] Человек {pid} вернулся ({now.strftime('%H:%M:%S')})")
                    info["in_frame"] = True

        # Проверяем, кто ушёл
        gone_ids = []
        for pid, info in list(tracked_people.items()):
            if pid not in current_ids:
                if time.time() - info["last_seen_time"] > lost_timeout:
                    duration = (datetime.now() - info["last_seen"]).total_seconds()
                    info["total_duration"] += duration
                    print(f"[DEPARTURE] Человек {pid} ушёл. Общее время: {info['total_duration']:.2f}s")
                    log_final_event(pid, info, csv_file)
                    gone_ids.append(pid)
                else:
                    info["in_frame"] = False
            else:
                info["total_duration"] = (now - info["first_arrival"]).total_seconds()

        for gid in gone_ids:
            tracked_people.pop(gid)
    except Exception as e:
        print(f"[TRACKING ERROR] {e}")
        traceback.print_exc()


# ---------- Отрисовка (оставлена для совместимости, но не вызывается) ----------
def draw_detections(frame, people):
    try:
        for p in people:
            x1, y1, x2, y2 = p["bbox"]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, p["center"], 5, color, -1)
            cv2.putText(frame, f"ID {p['track_id']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except Exception as e:
        print(f"[DRAW ERROR] {e}")


# ---------- Обработка кадра ----------
def process_frame(result, model, confidence, tracked_people, csv_file):
    try:
        # Копия кадра не нужна без отрисовки, но оставим для совместимости
        people = get_person_detections(result, model, confidence)
        update_tracked_people(people, tracked_people, csv_file)
        # draw_detections не вызывается — изображение не используется
        return people  # возвращаем только данные
    except Exception as e:
        print(f"[PROCESS ERROR] {e}")
        return []


# ---------- Потоки ----------
def frame_reader(camera_url, frame_queue, stop_event):
    cap = None
    while not stop_event.is_set():
        try:
            if cap is None or not cap.isOpened():
                print("[CONNECT] Подключение к камере...")
                cap = cv2.VideoCapture(camera_url)
                if not cap.isOpened():
                    print("[CAMERA ERROR] Не удалось подключиться. Повтор через 5 сек.")
                    time.sleep(5)
                    continue

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[FRAME ERROR] Ошибка чтения кадра, попытка повторить...")
                time.sleep(0.5)
                continue

            if not frame_queue.full():
                frame_queue.put(frame)

        except Exception as e:
            print(f"[READER ERROR] {e}")
            traceback.print_exc()
            time.sleep(5)

    if cap:
        cap.release()


def frame_processor(frame_queue, model, confidence, stop_event, csv_file):
    tracked_people = {}
    frame_count = 0
    start_time = time.time()
    report_interval = 10  # выводить FPS каждые N кадров

    while not stop_event.is_set():
        try:
            if frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = frame_queue.get()
            results = model.track(frame, persist=True)
            if not results:
                continue

            process_frame(results[0], model, confidence, tracked_people, csv_file)

            # Подсчёт FPS
            frame_count += 1
            if frame_count % report_interval == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[INFO] Средний FPS за последние {report_interval} кадров: {fps:.2f}")
                # Сброс счётчиков для плавного усреднения
                frame_count = 0
                start_time = time.time()

        except Exception as e:
            print(f"[PROCESSOR ERROR] {e}")
            traceback.print_exc()
            time.sleep(1)


# ---------- Основной запуск ----------
def main():
    try:
        config = load_config()
        confidence = config.get("confidence", 0.5)
        camera_url = config.get("camera_url")
        model_path = config.get("yolo_model", "yolo_model/yolo11n.pt")

        print("[INIT] Загрузка модели YOLO...")
        model = YOLO(model_path)
        csv_file = init_csv()

        frame_queue = queue.Queue(maxsize=5)
        stop_event = threading.Event()

        reader_thread = threading.Thread(target=frame_reader, args=(camera_url, frame_queue, stop_event))
        processor_thread = threading.Thread(target=frame_processor, args=(frame_queue, model, confidence, stop_event, csv_file))

        reader_thread.start()
        processor_thread.start()

        reader_thread.join()
        processor_thread.join()

    except KeyboardInterrupt:
        print("\n[STOP] Принудительная остановка пользователем.")
    except Exception as e:
        print(f"[MAIN ERROR] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()