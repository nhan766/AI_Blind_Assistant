# Cấu hình khớp nối cơ thể người (COCO Keypoints)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Đầu, mắt, tai
    (5, 6),                              # Vai
    (5, 7), (7, 9),                      # Tay trái (Vai -> Khuỷu -> Cổ tay)
    (6, 8), (8, 10),                     # Tay phải (Vai -> Khuỷu -> Cổ tay)
    (5, 11), (6, 12), (11, 12),          # Thân mình & Hông
    (11, 13), (13, 15),                  # Chân trái (Hông -> Gối -> Cổ chân)
    (12, 14), (14, 16)                   # Chân phải (Hông -> Gối -> Cổ chân)
]

# Thông số vận hành
VOICE_COOLDOWN = 3.5       # Thời gian nghỉ tối thiểu giữa 2 câu nói (giây)
SMOOTHING_FRAMES = 12      # Số frame lưu lại để lọc nhiễu hướng đi
DEFAULT_CONFIDENCE = 0.35  # Ngưỡng tin cậy của YOLO
DEFAULT_CAMERA = 0         # Cổng camera mặc định (0: webcam laptop)