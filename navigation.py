class NavigationEngine:
    def __init__(self):
        self.current_state = "GO_STRAIGHT"
        self.candidate_state = "GO_STRAIGHT"
        self.candidate_count = 0
        self.NORMAL_FRAMES = 8
        self.EMERGENCY_FRAMES = 3

    def analyze_frame(self, detections, frame_width, frame_height):
        l_bound = frame_width * 0.33
        r_bound = frame_width * 0.67
        center_mid = frame_width / 2

        zone_threats = {"LEFT": 0.0, "CENTER": 0.0, "RIGHT": 0.0}
        close_objects = []
        center_close_objs = []

        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            w_box = max(1.0, float(x2 - x1))
            h_box = float(y2 - y1)
            cx = (x1 + x2) / 2
            
            area_ratio = (w_box * h_box) / (frame_width * frame_height)
            height_ratio = h_box / frame_height

            # --- CHUẨN HÓA NGƯỠNG CỰ LY THỰC TẾ ---
            # 1. Rất gần (< 1.2m): Người choán gần hết chiều cao camera
            if area_ratio > 0.28 or height_ratio > 0.65:
                dist = "VERY_CLOSE"
                threat = 3.0
            # 2. Gần (1.5m - 2.5m): Cần chủ động chuyển hướng né
            elif area_ratio > 0.12 or height_ratio > 0.45:
                dist = "CLOSE"
                threat = 1.5
            # 3. An toàn (> 3m): Chưa gây nguy hiểm, không tính điểm đe dọa
            else:
                dist = "FAR"
                threat = 0.0

            overlap_l = max(0.0, min(float(x2), l_bound) - max(float(x1), 0.0))
            overlap_c = max(0.0, min(float(x2), r_bound) - max(float(x1), l_bound))
            overlap_r = max(0.0, min(float(x2), float(frame_width)) - max(float(x1), r_bound))

            zone_threats["LEFT"] += threat * (overlap_l / w_box)
            zone_threats["CENTER"] += threat * (overlap_c / w_box)
            zone_threats["RIGHT"] += threat * (overlap_r / w_box)

            primary_zone = "CENTER" if overlap_c >= max(overlap_l, overlap_r) else ("LEFT" if overlap_l > overlap_r else "RIGHT")
            
            if dist in ["VERY_CLOSE", "CLOSE"]:
                close_objects.append({"label": label, "zone": primary_zone, "distance": dist})
                if primary_zone == "CENTER" or overlap_c > 0:
                    center_close_objs.append(cx)

        c_threat = zone_threats["CENTER"]
        l_threat = zone_threats["LEFT"]
        r_threat = zone_threats["RIGHT"]

        left_safe = (l_threat < 1.0)
        right_safe = (r_threat < 1.0)

        # --- LOGIC ĐIỀU HƯỚNG ---
        # Chỉ can thiệp đổi hướng khi làn giữa có vật cản gần (threat >= 1.2)
        if c_threat >= 1.2:
            # Tình huống 1: Cả 2 bên đều bị chặn cứng -> BẮT BUỘC DỪNG
            if not left_safe and not right_safe:
                instant_cmd = "STOP"
            
            # Tình huống 2: Có lối thoát -> Ưu tiên né sang bên an toàn
            else:
                avg_cx = sum(center_close_objs) / len(center_close_objs) if center_close_objs else center_mid
                
                # Vật cản lệch TRÁI -> né PHẢI
                if avg_cx < center_mid:
                    if right_safe:
                        instant_cmd = "TURN_RIGHT"
                    elif left_safe:
                        instant_cmd = "TURN_LEFT"
                    else:
                        instant_cmd = "STOP"
                # Vật cản lệch PHẢI -> né TRÁI
                else:
                    if left_safe:
                        instant_cmd = "TURN_LEFT"
                    elif right_safe:
                        instant_cmd = "TURN_RIGHT"
                    else:
                        instant_cmd = "STOP"
        else:
            # Làn giữa còn khoảng cách an toàn -> TIẾP TỤC ĐI THẲNG
            instant_cmd = "GO_STRAIGHT"

        # --- BỘ LỌC CHỐNG RUNG HƯỚNG ---
        required_frames = self.EMERGENCY_FRAMES if instant_cmd == "STOP" else self.NORMAL_FRAMES
        if instant_cmd == self.candidate_state:
            self.candidate_count += 1
            if self.candidate_count >= required_frames:
                self.current_state = instant_cmd
        else:
            self.candidate_state = instant_cmd
            self.candidate_count = 1

        instructions = {
            "GO_STRAIGHT": "Phía trước an toàn, tiếp tục đi thẳng.",
            "TURN_LEFT": "Vật cản phía trước, hãy bước sang trái.",
            "TURN_RIGHT": "Vật cản phía trước, hãy bước sang phải.",
            "STOP": "Nguy hiểm phía trước! Dừng lại ngay."
        }

        alert_level = "CRITICAL" if self.current_state == "STOP" else ("WARNING" if "TURN" in self.current_state else "SAFE")

        return {
            "instruction": instructions[self.current_state],
            "direction": self.current_state,
            "alert_level": alert_level,
            "close_objects": close_objects
        }