import cv2
import time
from ultralytics import YOLO

# 1. Nạp mô hình đã train
model = YOLO("best.pt")

# Từ điển dịch các nhãn phổ biến sang tiếng Việt
LABEL_MAP = {
    "Animal": "Động vật",
    "Person": "Người",
    "Vehicle": "Xe cộ",
    "Pothole": "Ổ gà / Hố",
    "Pole": "Cột điện / Trụ",
    "Obstacle": "Vật cản",
    "Crosswalk": "Vạch qua đường",
    "Stairs": "Cầu thang"
}

# 2. Khởi tạo Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_alert = "Dang quet tim vat can..."
last_print_time = 0

print("--- HE THONG BAT DAU QUET (Nhan 'q' de thoat) ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    # Vẽ 2 đường kẻ chia khung hình làm 3 cột: Trái | Giữa | Phải để dễ quan sát
    cv2.line(frame, (w // 3, 0), (w // 3, h), (100, 100, 100), 1)
    cv2.line(frame, (2 * (w // 3), 0), (2 * (w // 3), h), (100, 100, 100), 1)

    # Dự đoán thời gian thực (tắt verbose để không in log rác)
    results = model(frame, stream=True, conf=0.45, verbose=False)

    detected_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            raw_name = model.names[cls_id]
            cls_name = LABEL_MAP.get(raw_name, raw_name)

            # Tính diện tích box so với toàn khung hình
            box_area = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / (w * h)

            # Xác định vị trí dựa trên tâm tọa độ X
            center_x = (x1 + x2) // 2
            if center_x < w // 3:
                direction = "Ben trai"
            elif center_x < 2 * (w // 3):
                direction = "Phia truoc"
            else:
                direction = "Ben phai"

            # Ước lượng mức độ gần/xa dựa trên kích thước khung nhận diện
            if area_ratio > 0.20:
                distance_status = "RAT GAN"
            elif area_ratio > 0.08:
                distance_status = "Gan"
            else:
                distance_status = "Xa"

            # Vẽ khung xanh quanh vật thể
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{cls_name} ({distance_status})"
            cv2.putText(frame, label_text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            detected_objects.append(f"{cls_name} o {direction} [{distance_status}]")

    # Nếu có vật cản, lấy thông tin cập nhật lên banner
    current_time = time.time()
    if detected_objects:
        current_alert = detected_objects[0]  # Lấy vật thể nổi bật nhất
        
        # In ra terminal mỗi 1 giây 1 lần để theo dõi mà không bị spam dòng
        if current_time - last_print_time > 1.0:
            print(" | ".join(detected_objects))
            last_print_time = current_time
    else:
        current_alert = "Duong thong thoang (Khong co vat can)"

    # Vẽ bảng hiển thị cảnh báo dạng chữ to nổi bật ở phía trên cùng khung hình
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, f">> {current_alert}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Test AI Nhan Dien - Blind Assistant", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()