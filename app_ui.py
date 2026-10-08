import os
import time
import threading
import queue
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

from config import SKELETON_CONNECTIONS, VOICE_COOLDOWN, DEFAULT_CONFIDENCE, DEFAULT_CAMERA
from voice import VoiceAssistant
from navigation import NavigationEngine

class BlindAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Blind Assistant - Trợ Lý Dẫn Đường")
        self.root.geometry("1100x720")
        self.root.configure(bg="#121212")

        self.is_running = True
        self.voice = VoiceAssistant(cooldown=VOICE_COOLDOWN)
        self.nav_engine = NavigationEngine()

        # Nạp mô hình Object & Pose
        self.model_path = os.path.join("weights", "best.pt")
        if not os.path.exists(self.model_path):
            self.model_path = "yolov8n.pt"
        self.model = YOLO(self.model_path)
        self.pose_model = YOLO("yolov8n-pose.pt")

        # Nguồn camera mặc định
        self.video_source = DEFAULT_CAMERA
        self.cap = cv2.VideoCapture(self.video_source)
        
        self.voice_enabled = tk.BooleanVar(value=True)
        self.frame_queue = queue.Queue(maxsize=1)
        self.latest_nav_result = None

        self._build_ui()

        # Phím tắt đóng ứng dụng
        self.root.bind('<Escape>', lambda e: self.on_closing())
        self.root.bind('q', lambda e: self.on_closing())

        # Khởi chạy luồng AI
        self.ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
        self.ai_thread.start()

        self._update_gui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#1F1F1F", height=50)
        header.pack(fill=tk.X, side=tk.TOP)
        tk.Label(
            header, text="HỆ THỐNG TRỢ LÝ HỖ TRỢ NGƯỜI KHIẾM THỊ",
            font=("Helvetica", 13, "bold"), fg="#00E5FF", bg="#1F1F1F"
        ).pack(pady=10)

        main_container = tk.Frame(self.root, bg="#121212")
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.video_frame = tk.Label(main_container, bg="#000000", bd=2, relief=tk.RIDGE)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

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

        # Các nút điều khiển
        tk.Button(
            sidebar, text="📁 Chọn File Video Để Test", font=("Helvetica", 10, "bold"),
            bg="#37474F", fg="#FFFFFF", relief=tk.FLAT, command=self.load_test_video
        ).pack(fill=tk.X, padx=15, pady=(15, 5))

        tk.Button(
            sidebar, text="📷 Quay Lại Camera Laptop (Cam 0)", font=("Helvetica", 10),
            bg="#263238", fg="#B0BEC5", relief=tk.FLAT, command=self.switch_to_camera
        ).pack(fill=tk.X, padx=15, pady=(0, 10))

        tk.Checkbutton(
            sidebar, text="Bật Giọng Nói Tiếng Việt", variable=self.voice_enabled,
            font=("Helvetica", 11), fg="#FFFFFF", bg="#1E1E1E", selectcolor="#2A2A2A"
        ).pack(pady=5)

        tk.Button(
            sidebar, text="THOÁT ỨNG DỤNG (Esc)", font=("Helvetica", 11, "bold"),
            bg="#D50000", fg="#FFFFFF", relief=tk.FLAT, command=self.on_closing
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=15)

    def load_test_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.video_source = file_path
            if self.cap.isOpened():
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_source)

    def switch_to_camera(self):
        self.video_source = DEFAULT_CAMERA
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        time.sleep(0.1)
        self.cap = cv2.VideoCapture(self.video_source)
        print("[*] Đã chuyển sang Webcam Laptop (Camera 0)")

    def _draw_hud(self, frame, nav_result):
        h, w, _ = frame.shape
        l_bound = int(w / 3)
        r_bound = int(2 * w / 3)

        cv2.line(frame, (l_bound, 0), (l_bound, h), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(frame, (r_bound, 0), (r_bound, h), (80, 80, 80), 1, cv2.LINE_AA)

        cv2.putText(frame, "TRAI", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        cv2.putText(frame, "GIUA", (l_bound + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "PHAI", (r_bound + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        if nav_result:
            alert = nav_result["alert_level"]
            if alert == "CRITICAL":
                cv2.rectangle(frame, (l_bound, 0), (r_bound, h), (0, 0, 255), 3)
            elif alert == "WARNING":
                cv2.rectangle(frame, (l_bound, 0), (r_bound, h), (0, 165, 255), 2)

        return frame

    def _ai_worker(self):
        frame_skip = 0
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.video_source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.02)
                continue

            # NẾU LÀ VIDEO: Nhảy cóc khung hình để video chạy mượt, đúng tốc độ thật
            if isinstance(self.video_source, str):
                frame_skip += 1
                if frame_skip % 2 != 0:  # Bỏ qua các frame lẻ, chỉ xử lý frame chẵn
                    continue

            # 1. Thu nhỏ khung hình về chiều rộng chuẩn 640px để tăng tốc độ xử lý gấp 3 lần
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)
                h, w = frame.shape[:2]

            if self.video_source == DEFAULT_CAMERA:
                frame = cv2.flip(frame, 1)

            # 2. Chạy YOLO Object Detection (thêm imgsz=384 để tăng tốc CPU)
            results = self.model(frame, verbose=False, conf=DEFAULT_CONFIDENCE, imgsz=384)[0]
            detections = []
            has_person = False

            for r in results.boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                cls_id = int(r.cls[0])
                label = self.model.names[cls_id]
                conf = float(r.conf[0])

                detections.append({"box": [x1, y1, x2, y2], "label": label, "conf": conf})

                if label.lower() == "person":
                    has_person = True
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(20, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 3. Chạy YOLO Pose chỉ khi có người (cũng thêm imgsz=384)
            if has_person:
                pose_results = self.pose_model(frame, verbose=False, conf=DEFAULT_CONFIDENCE, imgsz=384)[0]
                if pose_results.keypoints is not None:
                    for person_kpts in pose_results.keypoints:
                        kpts = person_kpts.xy[0].cpu().numpy()
                        confs = person_kpts.conf[0].cpu().numpy() if person_kpts.conf is not None else [1.0] * 17

                        for p1, p2 in SKELETON_CONNECTIONS:
                            if confs[p1] > 0.4 and confs[p2] > 0.4:
                                pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                                pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

                        for i, (kx, ky) in enumerate(kpts):
                            if confs[i] > 0.4 and kx > 0 and ky > 0:
                                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 255), -1, cv2.LINE_AA)

            # 4. Phân tích điều hướng & vẽ HUD
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