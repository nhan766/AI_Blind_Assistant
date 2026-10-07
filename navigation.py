class NavigationEngine:
    def __init__(self):
        self.current_state = "GO_STRAIGHT"
        self.candidate_state = "GO_STRAIGHT"
        self.candidate_count = 0
        self.NORMAL_FRAMES = 10
        self.EMERGENCY_FRAMES = 3

    def analyze_frame(self, detections, frame_width, frame_height):
        l_bound = frame_width * 0.33
        r_bound = frame_width * 0.67
        center_mid = frame_width / 2

        # Chỉ tính điểm đe dọa từ các vật thể ở cự ly GẦN hoặc RẤT GẦN
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

            # Phân loại cự ly
            if area_ratio > 0.20 or height_ratio > 0.50:
                dist = "VERY_CLOSE"
                threat = 3.0
            elif area_ratio > 0.07 or height_ratio > 0.28:
                dist = "CLOSE"
                threat = 1.5
            else:
                dist = "FAR"
                threat = 0.0  # Vật ở xa không tính điểm đe dọa để tránh cản trở lối né

            # Phân bổ độ phủ chiều ngang
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

        # --- RA QUYẾT ĐỊNH ĐIỀU HƯỚNG ---
        if c_threat >= 1.2:  # Làn giữa có vật cản gần
            # 1. Quá nguy hiểm hoặc cả 2 bên đều vướng vật cản gần -> DỪNG LẠI
            if c_threat >= 2.8 and (l_threat >= 1.2 or r_threat >= 1.2):
                instant_cmd = "STOP"
            
            # 2. Xét vị trí lệch của vật cản sát mặt
            else:
                # Tính tâm trung bình của vật cản phía trước
                avg_cx = sum(center_close_objs) / len(center_close_objs) if center_close_objs else center_mid
                
                # Nếu vật cản ở giữa lệch sang bên PHẢI (cx >= center_mid) -> BẮT BUỘC NÉ TRÁI
                if avg_cx >= center_mid:
                    if l_threat < 1.2:
                        instant_cmd = "TURN_LEFT"
                    else:
                        instant_cmd = "STOP"  # Bên trái vướng vật cản gần, không né được thì dừng
                
                # Nếu vật cản ở giữa lệch sang bên TRÁI (cx < center_mid) -> BẮT BUỘC NÉ PHẢI
                else:
                    if r_threat < 1.2:
                        instant_cmd = "TURN_RIGHT"
                    else:
                        instant_cmd = "STOP"

        else:
            instant_cmd = "GO_STRAIGHT"

        # --- BỘ LỌC CHỐNG NHẢY HƯỚNG ---
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