import os
import time
import threading
import queue
from collections import deque, Counter
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from gtts import gTTS
import pygame
from ultralytics import YOLO

# 1. TRỢ LÝ GIỌNG NÓI TIẾNG VIỆT 

class VoiceAssistant:
    def __init__(self, cooldown=3.5):
        self.cooldown = cooldown
        self.last_speech_time = 0
        self.last_message = ""
        self.msg_queue = queue.Queue()
        self.is_running = True
        
        pygame.mixer.init()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.is_running:
            try:
                text = self.msg_queue.get(timeout=0.2)
                if not self.is_running:
                    break
                
                # Tạo file âm thanh tạm thời
                tts = gTTS(text=text, lang='vi', slow=False)
                temp_file = f"temp_voice_{int(time.time()*1000)}.mp3"
                tts.save(temp_file)
                
                # Phát âm thanh
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and self.is_running:
                    time.sleep(0.05)
                
                pygame.mixer.music.unload()
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
                        
                self.msg_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Lỗi Voice]: {e}")

    def speak(self, text):
        current_time = time.time()
        # Chỉ nói khi khác câu trước hoặc đã qua thời gian cooldown
        if (text != self.last_message) and (current_time - self.last_speech_time > self.cooldown):
            self.last_message = text
            self.last_speech_time = current_time
            
            # Xóa các âm thanh cũ đang chờ
            while not self.msg_queue.empty():
                try:
                    self.msg_queue.get_nowait()
                except queue.Empty:
                    break
            self.msg_queue.put(text)

    def stop(self):
        self.is_running = False
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception:
            pass


# ==============================================================
# 2. LOGIC PHÂN TÍCH & BỘ LỌC CHỐNG RUNG (SMOOTHING ENGINE)
# ==============================================================
class NavigationEngine:
    def __init__(self):
        # Bộ đệm lưu 12 khung hình gần nhất để lọc nhiễu
        self.history = deque(maxlen=12)

    def analyze_frame(self, detections, frame_width, frame_height):
        left_boundary = frame_width / 3
        right_boundary = 2 * frame_width / 3

        zone_threats = {"LEFT": 0.0, "CENTER": 0.0, "RIGHT": 0.0}
        close_objects = []

        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            cx = (x1 + x2) / 2
            h = y2 - y1
            w = x2 - x1
            area_ratio = (w * h) / (frame_width * frame_height)

            if area_ratio > 0.22 or (h / frame_height) > 0.55:
                dist = "VERY_CLOSE"
                threat = 3.0
            elif area_ratio > 0.08 or (h / frame_height) > 0.30:
                dist = "CLOSE"
                threat = 1.5
            else:
                dist = "FAR"
                threat = 0.5

            if cx < left_boundary:
                zone = "LEFT"
            elif cx > right_boundary:
                zone = "RIGHT"
            else:
                zone = "CENTER"

            zone_threats[zone] += threat

            if dist in ["VERY_CLOSE", "CLOSE"]:
                close_objects.append({"label": label, "zone": zone, "distance": dist})

        center = zone_threats["CENTER"]
        left = zone_threats["LEFT"]
        right = zone_threats["RIGHT"]

        # Quyết định tạm thời cho khung hình hiện tại
        if center >= 1.5:
            if left < 1.0 and right < 1.0:
                raw_cmd = "TURN_LEFT" if left <= right else "TURN_RIGHT"
            elif left < 1.0:
                raw_cmd = "TURN_LEFT"
            elif right < 1.0:
                raw_cmd = "TURN_RIGHT"
            else:
                raw_cmd = "STOP"
        else:
            if left >= 2.0 and right < 1.0:
                raw_cmd = "SHIFT_RIGHT"
            elif right >= 2.0 and left < 1.0:
                raw_cmd = "SHIFT_LEFT"
            else:
                raw_cmd = "GO_STRAIGHT"

        # Đưa vào bộ đệm để lọc nhiễu
        self.history.append(raw_cmd)

        # Lấy quyết định chiếm số đông nhất trong 12 frames gần nhất
        most_common_cmd, count = Counter(self.history).most_common(1)[0]

        # Khẩu lệnh tương ứng
        instructions = {
            "GO_STRAIGHT": "Phía trước an toàn, tiếp tục đi thẳng.",
            "TURN_LEFT": "Vật cản phía trước, hãy bước sang trái.",
            "TURN_RIGHT": "Vật cản phía trước, hãy bước sang phải.",
            "SHIFT_LEFT": "Chú ý, hãy đi chếch sang trái.",
            "SHIFT_RIGHT": "Chú ý, hãy đi chếch sang phải.",
            "STOP": "Đường bị chặn hoàn toàn! Dừng lại ngay."
        }

        alert_level = "CRITICAL" if most_common_cmd == "STOP" else ("WARNING" if "TURN" in most_common_cmd else "SAFE")

        return {
            "instruction": instructions[most_common_cmd],
            "direction": most_common_cmd,
            "alert_level": alert_level,
            "close_objects": close_objects
        }


# ==============================================================
# 3. GIAO DIỆN CHÍNH
# ==============================================================
class BlindAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Blind Assistant - Trợ Lý Dẫn Đường")
        self.root.geometry("1100x720")
        self.root.configure(bg="#121212")

        self.is_running = True
        self.voice = VoiceAssistant(cooldown=3.5)
        self.nav_engine = NavigationEngine()

        # Nạp mô hình
        self.model_path = os.path.join("weights", "best.pt")
        if not os.path.exists(self.model_path):
            self.model_path = "yolov8n.pt"
        self.model = YOLO(self.model_path)

        self.video_source = 2  
        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        
        self.voice_enabled = tk.BooleanVar(value=True)
        self.frame_queue = queue.Queue(maxsize=1)
        self.latest_nav_result = None

        self._build_ui()

        # Phím tắt
        self.root.bind('<Escape>', lambda e: self.on_closing())
        self.root.bind('q', lambda e: self.on_closing())

        # Luồng AI ngầm
        self.ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
        self.ai_thread.start()

        self._update_gui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#1F1F1F", height=50)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header, text="HỆ THỐNG TRỢ LÝ HỖ TRỢ NGƯỜI KHIẾM THỊ",
            font=("Helvetica", 13, "bold"), fg="#00E5FF", bg="#1F1F1F"
        )
        title_lbl.pack(pady=10)

        main_container = tk.Frame(self.root, bg="#121212")
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Video Frame
        self.video_frame = tk.Label(main_container, bg="#000000", bd=2, relief=tk.RIDGE)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Sidebar
        sidebar = tk.Frame(main_container, bg="#1E1E1E", width=360)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="HƯỚNG ĐI ĐỀ XUẤT", font=("Helvetica", 11, "bold"), fg="#AAAAAA", bg="#1E1E1E").pack(pady=(15, 5))
        self.dir_icon_lbl = tk.Label(sidebar, text="↑", font=("Helvetica", 42, "bold"), fg="#00E676", bg="#1E1E1E")
        self.dir_icon_lbl.pack()
        self.dir_text_lbl = tk.Label(sidebar, text="ĐI THẲNG", font=("Helvetica", 16, "bold"), fg="#00E676", bg="#1E1E1E")
        self.dir_text_lbl.pack(pady=(0, 10))

        tk.Label(sidebar, text="CHỈ DẪN GIỌNG NÓI", font=("Helvetica", 11, "bold"), fg="#AAAAAA", bg="#1E1E1E").pack(pady=(10, 5))
        self.instruction_box = tk.Label(
            sidebar, text="Đang nhận dạng môi trường...", font=("Helvetica", 11),
            fg="#FFFFFF", bg="#2A2A2A", wraplength=320, height=3, relief=tk.GROOVE
        )
        self.instruction_box.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(sidebar, text="VẬT CẢN PHÁT HIỆN", font=("Helvetica", 11, "bold"), fg="#AAAAAA", bg="#1E1E1E").pack(pady=(15, 5))
        self.obj_listbox = tk.Listbox(sidebar, bg="#2A2A2A", fg="#00E5FF", font=("Consolas", 10), height=5, bd=0)
        self.obj_listbox.pack(fill=tk.X, padx=10, pady=5)

        # Nút chọn File Video để test
        video_btn = tk.Button(
            sidebar, text="📁 Chọn File Video Để Test", font=("Helvetica", 10, "bold"),
            bg="#37474F", fg="#FFFFFF", relief=tk.FLAT, command=self.load_test_video
        )
        video_btn.pack(fill=tk.X, padx=15, pady=(15, 5))

        cam_btn = tk.Button(
            sidebar, text="📷 Quay Lại Camera Trực Tiếp", font=("Helvetica", 10),
            bg="#263238", fg="#B0BEC5", relief=tk.FLAT, command=self.switch_to_camera
        )
        cam_btn.pack(fill=tk.X, padx=15, pady=(0, 10))

        voice_chk = tk.Checkbutton(
            sidebar, text="Bật Giọng Nói Tiếng Việt", variable=self.voice_enabled,
            font=("Helvetica", 11), fg="#FFFFFF", bg="#1E1E1E", selectcolor="#2A2A2A"
        )
        voice_chk.pack(pady=5)

        quit_btn = tk.Button(
            sidebar, text="THOÁT ỨNG DỤNG (Esc)", font=("Helvetica", 11, "bold"),
            bg="#D50000", fg="#FFFFFF", relief=tk.FLAT, command=self.on_closing
        )
        quit_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)

    def load_test_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.video_source = file_path
            if self.cap.isOpened():
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_source)

    def switch_to_camera(self):
            self.video_source = 1 
            if self.cap.isOpened():
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)

    def _draw_hud(self, frame, nav_result):
        h, w, _ = frame.shape
        l_bound = int(w / 3)
        r_bound = int(2 * w / 3)

        cv2.line(frame, (l_bound, 0), (l_bound, h), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(frame, (r_bound, 0), (r_bound, h), (80, 80, 80), 1, cv2.LINE_AA)

        cv2.putText(frame, "TRAI", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(frame, "GIUA (DI CHUYEN)", (l_bound + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "PHAI", (r_bound + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        if nav_result:
            alert = nav_result["alert_level"]
            if alert == "CRITICAL":
                cv2.rectangle(frame, (l_bound, 0), (r_bound, h), (0, 0, 255), 3)
            elif alert == "WARNING":
                cv2.rectangle(frame, (l_bound, 0), (r_bound, h), (0, 165, 255), 2)

        return frame

    def _ai_worker(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                # Nếu là video thì lặp lại từ đầu
                if isinstance(self.video_source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue

            if self.video_source == 0:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            # Chạy YOLO với conf=0.35 để giảm nhận diện rác
            results = self.model(frame, verbose=False, conf=0.35)[0]
            detections = []

            for r in results.boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                cls_id = int(r.cls[0])
                label = self.model.names[cls_id]
                conf = float(r.conf[0])

                detections.append({"box": [x1, y1, x2, y2], "label": label, "conf": conf})

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            nav_result = self.nav_engine.analyze_frame(detections, w, h)
            frame = self._draw_hud(frame, nav_result)

            if self.voice_enabled.get():
                self.voice.speak(nav_result["instruction"])

            self.latest_nav_result = nav_result
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def _update_gui(self):
        if not self.is_running:
            return

        try:
            frame = self.frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((700, 525), Image.Resampling.BILINEAR)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

            if self.latest_nav_result:
                self._update_sidebar(self.latest_nav_result)
        except queue.Empty:
            pass

        if self.is_running:
            self.root.after(20, self._update_gui)

    def _update_sidebar(self, nav):
        self.instruction_box.config(text=nav["instruction"])

        cmd = nav["direction"]
        if cmd == "GO_STRAIGHT":
            self.dir_icon_lbl.config(text="↑", fg="#00E676")
            self.dir_text_lbl.config(text="ĐI THẲNG", fg="#00E676")
        elif cmd in ["TURN_LEFT", "SHIFT_LEFT"]:
            self.dir_icon_lbl.config(text="←", fg="#FFD600")
            self.dir_text_lbl.config(text="RẼ TRÁI", fg="#FFD600")
        elif cmd in ["TURN_RIGHT", "SHIFT_RIGHT"]:
            self.dir_icon_lbl.config(text="→", fg="#FFD600")
            self.dir_text_lbl.config(text="RẼ PHẢI", fg="#FFD600")
        elif cmd == "STOP":
            self.dir_icon_lbl.config(text="✖", fg="#FF1744")
            self.dir_text_lbl.config(text="DỪNG LẠI", fg="#FF1744")

        self.obj_listbox.delete(0, tk.END)
        for obj in nav["close_objects"]:
            zone_vn = {"LEFT": "Trái", "CENTER": "Giữa", "RIGHT": "Phải"}.get(obj["zone"])
            dist_vn = {"VERY_CLOSE": "RẤT GẦN", "CLOSE": "Gần", "FAR": "Xa"}.get(obj["distance"])
            self.obj_listbox.insert(tk.END, f"• {obj['label']} [{zone_vn} - {dist_vn}]")

    def on_closing(self):
        self.is_running = False
        try:
            self.voice.stop()
        except Exception:
            pass
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = BlindAssistantApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()