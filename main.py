import tkinter as tk
from app_ui import BlindAssistantApp

def main():
    root = tk.Tk()
    app = BlindAssistantApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()