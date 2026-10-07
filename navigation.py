class NavigationEngine:
    def __init__(self):
        self.current_state = "GO_STRAIGHT"
        self.candidate_state = "GO_STRAIGHT"
        self.candidate_count = 0
        
        # Bộ đệm thích ứng: STOP phản ứng nhanh, RẼ/THẲNG giữ ổn định
        self.NORMAL_FRAMES = 10
        self.EMERGENCY_FRAMES = 3

    def analyze_frame(self, detections, frame_width, frame_height):
        # 1. Định nghĩa ranh giới 3 làn (Trái: 0-33%, Giữa: 33-67%, Phải: 67-100%)
        l_bound = frame_width * 0.33
        r_bound = frame_width * 0.67

        zone_threats = {"LEFT": 0.0, "CENTER": 0.0, "RIGHT": 0.0}
        close_objects = []

        # 2. Phân tích từng vật thể và phân bổ độ phủ sang các làn
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            w_box = max(1.0, float(x2 - x1))
            h_box = float(y2 - y1)
            area_ratio = (w_box * h_box) / (frame_width * frame_height)
            height_ratio = h_box / frame_height

            # Tính điểm đe dọa theo cự ly
            if area_ratio > 0.22 or height_ratio > 0.55:
                dist = "VERY_CLOSE"
                threat = 3.0
            elif area_ratio > 0.08 or height_ratio > 0.30:
                dist = "CLOSE"
                threat = 1.5
            else:
                dist = "FAR"
                threat = 0.5

            # Tính độ rộng vật thể nằm trên từng làn (tránh lỗi nhảy làn ở mép)
            overlap_l = max(0.0, min(float(x2), l_bound) - max(float(x1), 0.0))
            overlap_c = max(0.0, min(float(x2), r_bound) - max(float(x1), l_bound))
            overlap_r = max(0.0, min(float(x2), float(frame_width)) - max(float(x1), r_bound))

            ratio_l = overlap_l / w_box
            ratio_c = overlap_c / w_box
            ratio_r = overlap_r / w_box

            # Phân bổ độ nguy hiểm theo tỷ lệ thực tế
            zone_threats["LEFT"] += threat * ratio_l
            zone_threats["CENTER"] += threat * ratio_c
            zone_threats["RIGHT"] += threat * ratio_r

            # Xác định làn chính của vật thể để hiển thị log lên UI
            primary_zone = "CENTER" if ratio_c >= max(ratio_l, ratio_r) else ("LEFT" if ratio_l > ratio_r else "RIGHT")
            if dist in ["VERY_CLOSE", "CLOSE"]:
                close_objects.append({"label": label, "zone": primary_zone, "distance": dist})

        c_threat = zone_threats["CENTER"]
        l_threat = zone_threats["LEFT"]
        r_threat = zone_threats["RIGHT"]

        # Ngưỡng an toàn: < 0.8 là thông thoáng; >= 1.2 là bị vướng
        SAFE_LIMIT = 0.8
        BLOCKED_LIMIT = 1.2

        # 3. Ma trận ra quyết định hướng đi an toàn
        if c_threat >= BLOCKED_LIMIT:
            left_is_safe = (l_threat < SAFE_LIMIT)
            right_is_safe = (r_threat < SAFE_LIMIT)

            # Trường hợp 1: Sát trước mặt cực kỳ nguy hiểm, hoặc cả 3 hướng đều bị chặn
            if c_threat >= 2.8 or (not left_is_safe and not right_is_safe):
                instant_cmd = "STOP"

            # Trường hợp 2: Cả 2 bên Trái và Phải đều trống
            elif left_is_safe and right_is_safe:
                if abs(l_threat - r_threat) > 0.2:
                    # Ưu tiên bên có ít vật thể/áp lực hơn
                    instant_cmd = "TURN_LEFT" if l_threat < r_threat else "TURN_RIGHT"
                else:
                    # Nếu độ thoáng tương đương: kiểm tra vật cản ở giữa lệch bên nào để né sang hướng ngược lại
                    center_objs = [d for d in detections if (d['box'][0] + d['box'][2]) / 2 >= l_bound and (d['box'][0] + d['box'][2]) / 2 <= r_bound]
                    if center_objs:
                        avg_cx = sum((d['box'][0] + d['box'][2]) / 2 for d in center_objs) / len(center_objs)
                        instant_cmd = "TURN_RIGHT" if avg_cx < (frame_width / 2) else "TURN_LEFT"
                    else:
                        instant_cmd = "TURN_LEFT"

            # Trường hợp 3: Chỉ có bên Trái an toàn
            elif left_is_safe:
                instant_cmd = "TURN_LEFT"

            # Trường hợp 4: Chỉ có bên Phải an toàn
            elif right_is_safe:
                instant_cmd = "TURN_RIGHT"

            # Trường hợp 5: Hai bên đều có vật cản ở mức trung bình -> Dừng lại để an toàn
            else:
                instant_cmd = "STOP"

        else:
            # Làn giữa hoàn toàn thông thoáng
            instant_cmd = "GO_STRAIGHT"

        # 4. Bộ lọc thích ứng (Debounce Filter)
        required_frames = self.EMERGENCY_FRAMES if instant_cmd == "STOP" else self.NORMAL_FRAMES

        if instant_cmd == self.candidate_state:
            self.candidate_count += 1
            if self.candidate_count >= required_frames:
                self.current_state = instant_cmd
        else:
            self.candidate_state = instant_cmd
            self.candidate_count = 1

        # 5. Khẩu lệnh và mức độ cảnh báo
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