import os
import time
import threading
import queue
from gtts import gTTS
import pygame

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
                
                temp_file = f"temp_voice_{int(time.time()*1000)}.mp3"
                tts = gTTS(text=text, lang='vi', slow=False)
                tts.save(temp_file)
                
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
        if (text != self.last_message) and (current_time - self.last_speech_time > self.cooldown):
            self.last_message = text
            self.last_speech_time = current_time
            
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