import cv2
import time

# RTSP-поток
rtsp_url = "rtsp://admin:8PykD48PykD4!@172.20.2.229:554"

cap = cv2.VideoCapture(rtsp_url)

# Переменная для хранения координат
cursor_pos = (0, 0)

# Функция обратного вызова мыши
def mouse_callback(event, x, y, flags, param):
    global cursor_pos
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_pos = (x, y)

# Создаём окно и привязываем коллбэк
cv2.namedWindow("RTSP")
cv2.setMouseCallback("RTSP", mouse_callback)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка при получении кадра")
        continue

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Отобразим координаты на кадре
    cv2.putText(frame, f"X: {cursor_pos[0]} Y: {cursor_pos[1]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("RTSP", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
