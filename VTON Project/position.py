import cv2
import mediapipe as mp
import time
import math
import os

# --- CONFIGURATION ---
SAVE_DIR = "captured_poses"
REQUIRED_LANDMARKS = [11, 12, 23, 24] # Shoulders and Hips
VISIBILITY_THRESHOLD = 0.7
STABILITY_DURATION = 2.0 # Seconds to hold pose before snap

# Ratios to determine pose orientation
# (Shoulder Width / Torso Height)
# Front view: Shoulders are wide relative to height (Ratio > 0.5 approx)
# Side view: Shoulders are narrow (Ratio < 0.4 approx)
FRONT_RATIO_THRESH = 0.5 
SIDE_RATIO_THRESH = 0.35

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- HELPER FUNCTIONS ---

def are_points_in_frame(landmarks, width, height):
    """Checks if required points are visible and within image boundaries."""
    for idx in REQUIRED_LANDMARKS:
        lm = landmarks[idx]
        # Check visibility
        if lm.visibility < VISIBILITY_THRESHOLD:
            return False
        # Check boundaries (0.0 to 1.0) with a small buffer
        if not (0.05 < lm.x < 0.95 and 0.05 < lm.y < 0.95):
            return False
    return True

def get_pose_orientation(landmarks):
    """Calculates shoulder width vs torso height to guess orientation."""
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_hip = landmarks[23]
    r_hip = landmarks[24]

    # Calculate shoulder width (Horizontal distance)
    shoulder_width = abs(l_sh.x - r_sh.x)
    
    # Calculate Torso Height (Vertical distance avg)
    mid_shoulder_y = (l_sh.y + r_sh.y) / 2
    mid_hip_y = (l_hip.y + r_hip.y) / 2
    torso_height = abs(mid_hip_y - mid_shoulder_y)

    if torso_height == 0: return "UNKNOWN"

    ratio = shoulder_width / torso_height

    if ratio > FRONT_RATIO_THRESH:
        return "FRONT_OR_BACK" # Wide shoulders
    elif ratio < SIDE_RATIO_THRESH:
        return "SIDE" # Narrow shoulders
    else:
        return "TRANSITION" # Somewhere in between

# --- MAIN LOOP ---

cap = cv2.VideoCapture(0)

capture_stage = 0 # 0: Front, 1: Side, 2: Back, 3: Done
stages = ["FRONT VIEW", "SIDE VIEW", "BACK VIEW"]
filenames = ["front.jpg", "side.jpg", "back.jpg"]

pose_start_time = None # Timer for stability

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Flip for selfie mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = pose.process(rgb_frame)
        
        # Default status message
        status_text = "Looking for user..."
        color = (0, 255, 255) # Yellow (Waiting)

        if capture_stage < 3:
            target_view = stages[capture_stage]
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # 1. Check if whole body is in frame
                if are_points_in_frame(lm, w, h):
                    
                    # 2. Check Orientation
                    orientation = get_pose_orientation(lm)
                    
                    # Logic to match stage requirements
                    match_found = False
                    if capture_stage == 0 and orientation == "FRONT_OR_BACK": # Needs Front
                        match_found = True
                    elif capture_stage == 1 and orientation == "SIDE": # Needs Side
                        match_found = True
                    elif capture_stage == 2 and orientation == "FRONT_OR_BACK": # Needs Back
                        match_found = True
                    
                    if match_found:
                        if pose_start_time is None:
                            pose_start_time = time.time()
                        
                        # Calculate elapsed time
                        elapsed = time.time() - pose_start_time
                        countdown = max(0, STABILITY_DURATION - elapsed)
                        
                        if countdown == 0:
                            # --- SNAPSHOT ---
                            save_path = os.path.join(SAVE_DIR, filenames[capture_stage])
                            # Save the clean frame (without drawings)
                            cv2.imwrite(save_path, frame)
                            print(f"Captured {target_view}!")
                            
                            capture_stage += 1
                            pose_start_time = None
                            # Small freeze to show success
                            cv2.rectangle(frame, (0,0), (w,h), (255,255,255), 10)
                            cv2.waitKey(200) 
                        else:
                            status_text = f"Hold Still: {countdown:.1f}s"
                            color = (0, 255, 0) # Green (Good)
                            
                    else:
                        # In frame, but wrong angle
                        pose_start_time = None # Reset timer
                        status_text = f"Please turn to {target_view}"
                        color = (0, 0, 255) # Red
                else:
                    pose_start_time = None
                    status_text = "Step back / Center body"
                    color = (0, 0, 255)

                # Draw Landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        else:
            status_text = "All Done! Press Q to exit."
            color = (255, 215, 0) # Gold

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Step: {stages[min(capture_stage, 2)]}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Smart Body Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()