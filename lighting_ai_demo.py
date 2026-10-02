import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# --- DOWNLOAD MEDIAPIPE MODEL ---
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe model asset...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

GRID_RES = 100  # 100x100 resolution (10,000 LED nodes)

# --- HIGH-DENSITY 100x100 LED MATRIX RASTERIZATION ---
def render_skeleton_to_100x100(hand_landmarks_list, color=(0, 255, 120)):
    """Dynamically projects 3D hand coordinates onto a 100x100 addressable LED matrix."""
    matrix = np.zeros((GRID_RES, GRID_RES, 3), dtype=np.uint8)
    
    if not hand_landmarks_list:
        cv2.circle(matrix, (GRID_RES // 2, GRID_RES // 2), 6, (35, 35, 35), 1)
        return matrix
        
    for hand_lms in hand_landmarks_list:
        pts = [(int(np.clip(lm.x * (GRID_RES - 1), 0, GRID_RES - 1)), 
                int(np.clip(lm.y * (GRID_RES - 1), 0, GRID_RES - 1))) for lm in hand_lms]
        
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            cv2.line(matrix, pts[p1_idx], pts[p2_idx], color, 2)
            
        for pt in pts:
            cv2.circle(matrix, pt, 2, (255, 255, 255), -1)
            
    return matrix


# --- GESTURE CLASSIFIER ---
def get_finger_states(lms):
    thumb = abs(lms[4].x - lms[9].x) > abs(lms[3].x - lms[9].x)
    index = lms[8].y < lms[6].y
    middle = lms[12].y < lms[10].y
    ring = lms[16].y < lms[14].y
    pinky = lms[20].y < lms[18].y
    return [thumb, index, middle, ring, pinky]

def classify_gesture(results):
    if not results.hand_landmarks: return "NONE"
    if len(results.hand_landmarks) >= 2: return "DUAL_HANDS_ACTIVE"

    states = get_finger_states(results.hand_landmarks[0])
    
    if states == [True, True, True, True, True]: return "OPEN_PALM (FULL FLOOD)"
    if states == [False, True, True, True, True]: return "FOUR_FINGERS (THUMB TUCKED)"
    if states == [False, False, False, False, False]: return "CLOSED_FIST (STANDBY)"
    if states == [True, False, False, False, False]: return "THUMBS_UP (CONFIRMATION)"
    if states == [False, True, False, False, False]: return "INDEX_POINT (TARGETING)"
    if states == [False, True, True, False, False]: return "PEACE_SIGN (SPLIT SECTOR)"
    if states == [False, False, True, False, False]: return "MIDDLE_FINGER (ALERT_RED)"
    if states == [True, True, False, False, True]: return "SPIDERMAN (DUAL_BEAM)"
    if states == [False, True, False, False, True]: return "ROCK_ON (PERIMETER)"
    
    return f"CUSTOM_POSE ({sum(states)} FINGERS)"

def get_color_for_gesture(gesture):
    if "OPEN_PALM" in gesture: return (255, 255, 255)
    if "CLOSED_FIST" in gesture: return (0, 140, 255)
    if "INDEX_POINT" in gesture: return (255, 255, 0)
    if "PEACE_SIGN" in gesture: return (0, 255, 120)
    if "MIDDLE_FINGER" in gesture: return (0, 0, 255)
    if "SPIDERMAN" in gesture: return (255, 0, 180)
    if "DUAL_HANDS" in gesture: return (255, 200, 0)
    return (0, 230, 255)


# --- MAIN APPLICATION DASHBOARD ---
class LightingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bosch Capstone: VLA vs VLM Concurrent Testbed (100x100 Matrix)")
        self.root.geometry("1440x920")
        self.root.configure(bg="#121212")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#121212", foreground="white", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#4eb3f3")
        style.configure("Spec.TLabel", font=("Consolas", 8), foreground="#90a4ae", background="#222222")

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2)
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.cap = None
        self.is_running = True
        
        self.vlm_latency_ms = 350
        self.vlm_queue = []
        
        self.build_ui()
        self.start_camera(0)
        self.update_frame()

    def build_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Left Column (Camera Viewport)
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        ttk.Label(left_frame, text="Real-Time Computer Vision Feed", style="Header.TLabel").pack(anchor=tk.W)
        self.video_canvas = tk.Canvas(left_frame, width=640, height=480, bg="black", highlightthickness=1, highlightbackground="#333")
        self.video_canvas.pack(pady=10)
        
        cam_frame = ttk.Frame(left_frame)
        cam_frame.pack(fill=tk.X)
        ttk.Label(cam_frame, text="Active Video Sensor:").pack(side=tk.LEFT, padx=(0, 10))
        self.cam_combo = ttk.Combobox(cam_frame, values=[f"Camera {i}" for i in range(4)], state="readonly", width=12)
        self.cam_combo.current(0)
        self.cam_combo.pack(side=tk.LEFT)
        self.cam_combo.bind("<<ComboboxSelected>>", self.change_camera)

        self.lbl_status = ttk.Label(left_frame, text="Detected Pose: NONE", font=("Segoe UI", 15, "bold"), foreground="#ffd54f")
        self.lbl_status.pack(anchor=tk.W, pady=15)

        # Right Column (Dual 100x100 LED Grids)
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # USE GRID INSTEAD OF PACK FOR STRICT 50/50 SIZING
        dash_container = ttk.Frame(right_frame)
        dash_container.pack(fill=tk.BOTH, expand=True)
        dash_container.columnconfigure(0, weight=1, uniform="dash")
        dash_container.columnconfigure(1, weight=1, uniform="dash")
        dash_container.rowconfigure(0, weight=1)

        # --- VLA DASHBOARD ---
        vla_frame = tk.Frame(dash_container, bg="#1e1e1e", bd=1, relief=tk.SOLID)
        vla_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        vla_frame.pack_propagate(False) # Prevents child widgets from resizing the frame
        
        ttk.Label(vla_frame, text=" End-to-End VLA (Low Latency)", style="Header.TLabel", background="#1e1e1e", foreground="#ffb74d").pack(anchor=tk.W, pady=6)
        ttk.Label(vla_frame, text=" Model: Simlingo / OpenVLA (~2B - 7B Params)\n Target Hardware: NVIDIA Jetson AGX Orin\n Execution: Single-Pass Direct Action Head\n Inference Latency: ~10 - 20 ms", style="Spec.TLabel").pack(anchor=tk.W, padx=6, fill=tk.X)
        
        self.lbl_vla_lat = ttk.Label(vla_frame, text="Active Latency: 12.0 ms (Real-Time)", font=("Segoe UI", 10, "bold"), foreground="#66bb6a", background="#1e1e1e")
        self.lbl_vla_lat.pack(anchor=tk.W, padx=6, pady=6)

        ttk.Label(vla_frame, text="100x100 Matrix Output (10,000 Addressable Diodes)", style="Spec.TLabel").pack(anchor=tk.W, padx=6)
        self.vla_img_label = tk.Label(vla_frame, bg="#050505", highlightthickness=1, highlightbackground="#ffb74d")
        self.vla_img_label.pack(padx=6, pady=8, fill=tk.BOTH, expand=True)

        # --- MODULAR VLM DASHBOARD ---
        vlm_frame = tk.Frame(dash_container, bg="#1e1e1e", bd=1, relief=tk.SOLID)
        vlm_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        vlm_frame.pack_propagate(False) # Prevents child widgets from resizing the frame
        
        ttk.Label(vlm_frame, text=" Modular VLM + LLM (Decoupled)", style="Header.TLabel", background="#1e1e1e", foreground="#4fc3f7").pack(anchor=tk.W, pady=6)
        ttk.Label(vlm_frame, text=" Model: Decoupled Vision Backbone + Llama-3 / Phi-3\n Target Hardware: Host PC + Co-Processor\n Execution: Vision -> Scene Tokens -> LLM Reasoning\n Pipeline Latency: ~250 - 500 ms (Token Bottleneck)", style="Spec.TLabel").pack(anchor=tk.W, padx=6, fill=tk.X)
        
        self.lbl_vlm_lat = ttk.Label(vlm_frame, text="Active Latency: ~350.0 ms (Buffered)", font=("Segoe UI", 10, "bold"), foreground="#ef5350", background="#1e1e1e")
        self.lbl_vlm_lat.pack(anchor=tk.W, padx=6, pady=6)

        ttk.Label(vlm_frame, text="100x100 Matrix Output (Buffered Downstream Feed)", style="Spec.TLabel").pack(anchor=tk.W, padx=6)
        self.vlm_img_label = tk.Label(vlm_frame, bg="#050505", highlightthickness=1, highlightbackground="#4fc3f7")
        self.vlm_img_label.pack(padx=6, pady=8, fill=tk.BOTH, expand=True)

    def start_camera(self, cam_idx):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(cam_idx)
    
    def change_camera(self, event):
        idx = int(self.cam_combo.get().split(" ")[1])
        self.start_camera(idx)

    def render_matrix_to_label(self, label, matrix):
        # Lock target dimension to prevent UI shift
        target_dim = 340 
        upscaled = cv2.resize(matrix, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)
        img = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label.configure(image=imgtk)
        label.image = imgtk

    def update_frame(self):
        if not self.is_running: return
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            
            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    for connection in HAND_CONNECTIONS:
                        p1, p2 = hand_lms[connection[0]], hand_lms[connection[1]]
                        cv2.line(rgb_frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 255, 0), 2)
            
            gesture = classify_gesture(result)
            color = get_color_for_gesture(gesture)
            self.lbl_status.config(text=f"Detected Pose: {gesture}")

            lms_list = [h for h in result.hand_landmarks] if result.hand_landmarks else []
            vla_matrix = render_skeleton_to_100x100(lms_list, color=color)
            self.render_matrix_to_label(self.vla_img_label, vla_matrix)

            now = time.time()
            self.vlm_queue.append((now + (self.vlm_latency_ms / 1000.0), vla_matrix.copy()))
            
            while self.vlm_queue and now >= self.vlm_queue[0][0]:
                _, delayed_matrix = self.vlm_queue.pop(0)
                self.render_matrix_to_label(self.vlm_img_label, delayed_matrix)

            img = Image.fromarray(rgb_frame).resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.video_canvas.image = imgtk

        self.root.after(10, self.update_frame)
        
    def on_close(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LightingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()