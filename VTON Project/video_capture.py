import sys
import io

# --- FIX FOR WINDOWS EMOJI CRASH ---
# Forces UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
import mediapipe as mp
import math
import os
import time

# --- CONFIGURATION (Adjust these values) ---
OUTPUT_DIR = "captured_video"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. DISTANCE TARGET (Must be consistent)
TARGET_DISTANCE_CM = 300.0 

# 2. CAMERA CALIBRATION
FOCAL_LENGTH = 650.0 
AVG_SHOULDER_WIDTH_CM = 45.0 

# 3. CAPTURE SETTINGS
TOLERANCE_PX = 10 
RECORDING_DURATION = 15.0 # Seconds to record the full 360 rotation

# --- HELPER FUNCTIONS ---

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two (x,y) points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_target_pixel_size(real_width_cm, distance_cm, focal_length):
    """Target Px = (Focal Length * Real Width) / Distance"""
    if distance_cm == 0: return 0
    return (focal_length * real_width_cm) / distance_cm

# --- MAIN LOGIC ---

# 0. Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# 1. State Variables
video_writer = None
recording_start_time = None
TARGET_PX_SIZE = get_target_pixel_size(AVG_SHOULDER_WIDTH_CM, TARGET_DISTANCE_CM, FOCAL_LENGTH)

print(f"Status: Ready. Target Shoulder Size: {TARGET_PX_SIZE:.0f} pixels.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    h, w, _ = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 # Use actual FPS or default to 30
    
    # Flip for mirror view and process pose
    image = cv2.flip(frame, 1)
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    status_text = f"Stand {TARGET_DISTANCE_CM:.0f}cm away."
    status_color = (0, 255, 255)
    
    pixel_shoulder_width = 0.0
    is_at_target_distance = False

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        def get_pt(idx): return (int(lm[idx].x * w), int(lm[idx].y * h))
        
        # --- 1. HORIZONTAL MEASUREMENTS (For Distance Check) ---
        p_shoulder_l = get_pt(11)
        p_shoulder_r = get_pt(12)
        pixel_shoulder_width = calculate_distance(p_shoulder_l, p_shoulder_r)
        
        if abs(pixel_shoulder_width - TARGET_PX_SIZE) < TOLERANCE_PX:
            is_at_target_distance = True
            
    # --- 2. RECORDING LOGIC ---
    
    if not is_at_target_distance and video_writer is None:
        # User is NOT at the correct distance
        recording_start_time = None
        if pixel_shoulder_width > 0:
            diff = pixel_shoulder_width - TARGET_PX_SIZE
            if diff > 0:
                status_text = f"MOVE BACK ({abs(diff):.0f} px too large)"
                status_color = (0, 0, 255)
            else:
                status_text = f"MOVE FORWARD ({abs(diff):.0f} px too small)"
                status_color = (255, 0, 0)
    
    elif is_at_target_distance and video_writer is None:
        # User is at distance, ready to start recording
        if recording_start_time is None:
            recording_start_time = time.time()
            
        elapsed_ready = time.time() - recording_start_time
        
        if elapsed_ready >= 1.0: # Wait 1 sec before starting video
            
            # --- START RECORDING ---
            capture_timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(OUTPUT_DIR, f"360_scan_{capture_timestamp}.avi")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec for AVI
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (w, h))
            print(f"\n🎬 Starting recording: {video_filename}")
            
        else:
            status_text = f"Ready to scan... {int(1.0 - elapsed_ready + 1)}"
            status_color = (0, 255, 0)

    # --- 3. ACTIVE RECORDING STATE ---
    if video_writer is not None:
        elapsed_recording = time.time() - recording_start_time
        countdown = max(0, RECORDING_DURATION - elapsed_recording)
        
        if countdown > 0:
            # Write the current frame to the video file
            video_writer.write(image)
            status_text = f"ROTATING... {countdown:.1f}s Left. Rotate SLOWLY 360°"
            status_color = (0, 255, 0)
            
            # Draw a recording indicator
            cv2.circle(image, (w - 30, 30), 10, (0, 0, 255), -1)
        else:
            # --- STOP RECORDING ---
            print(" Recording finished. Video saved.")
            video_writer.release()
            video_writer = None
            recording_start_time = None
            # Display flash and pause
            cv2.putText(image, "SCAN COMPLETE!", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
            cv2.imshow('360 Scan Capture', image)
            cv2.waitKey(2000) # Show completion message for 2 seconds
            break # Exit loop after completion

    # --- FINAL DRAWING OVERLAYS ---
    cv2.putText(image, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(image, f"Target Size: {TARGET_PX_SIZE:.0f} px", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('360 Scan Capture', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("\n--- Capture Session Complete ---")
if video_writer is not None:
    video_writer.release() # Ensure writer is closed if loop was broken
    print("Video writer closed.")

cap.release()
cv2.destroyAllWindows()