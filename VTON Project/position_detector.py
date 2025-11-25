import cv2
import mediapipe as mp
import time
import os

# --- CONFIGURATION ---
SAVE_DIR = "captured_poses"
# We want high confidence (visibility) so we don't capture blurry/bad tracking
VISIBILITY_THRESH = 0.8 
STABILITY_DURATION = 1.5 # Seconds to hold pose

# Setup MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def check_points_visibility(landmarks, required_indices):
    """
    Checks if specific body parts are visible and inside the screen.
    Unlike before, this allows you to be anywhere in the frame (0.0 to 1.0).
    """
    for idx in required_indices:
        lm = landmarks[idx]
        
        # 1. Check if the model is confident (it sees the point clearly)
        if lm.visibility < VISIBILITY_THRESH:
            return False
            
        # 2. Check if point is strictly inside the image frame
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            return False
            
    return True

def detect_orientation(landmarks):
    """
    Uses the Z-coordinate (Depth) to see which shoulder is closer.
    Negative Z = Closer to camera.
    """
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    
    # Calculate the difference in depth
    # If diff is small, you are facing front.
    # If R is much closer (negative), it's Right Side.
    z_diff = l_shoulder.z - r_shoulder.z
    
    # Thresholds for rotation
    if z_diff > 0.15: # Left shoulder is much further back -> Right Side is forward
        return "RIGHT_SIDE"
    elif z_diff < -0.15: # Right shoulder is much further back -> Left Side is forward
        return "LEFT_SIDE"
    else:
        return "FRONT"

# --- MAIN EXECUTION ---

cap = cv2.VideoCapture(0)

# Sequence: Front -> Right Profile -> Left Profile
stages = ["FRONT", "RIGHT_SIDE", "LEFT_SIDE"]
filenames = ["front.jpg", "right_side.jpg", "left_side.jpg"]
current_stage_idx = 0

pose_start_time = None

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Flip frame horizontally for "Mirror" feel
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        results = pose.process(rgb_frame)
        
        status_msg = "Looking for user..."
        color = (0, 255, 255) # Yellow (Waiting)

        if current_stage_idx < len(stages):
            target_view = stages[current_stage_idx]
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # 1. Determine which points we need to see based on stage
                if target_view == "FRONT":
                    # Need both shoulders (11,12) and hips (23,24)
                    req_points = [11, 12, 23, 24]
                elif target_view == "RIGHT_SIDE":
                    # Need Right Shoulder(12), Right Hip(24)
                    req_points = [12, 24]
                elif target_view == "LEFT_SIDE":
                    # Need Left Shoulder(11), Left Hip(23)
                    req_points = [11, 23]
                
                # 2. Check Visibility & Orientation
                is_visible = check_points_visibility(lm, req_points)
                current_orientation = detect_orientation(lm)
                
                # Logic: Are points clear AND is orientation correct?
                if is_visible and current_orientation == target_view:
                    
                    # Start Timer
                    if pose_start_time is None:
                        pose_start_time = time.time()
                    
                    elapsed = time.time() - pose_start_time
                    countdown = max(0, STABILITY_DURATION - elapsed)
                    
                    if countdown == 0:
                        # --- SNAP ---
                        save_path = os.path.join(SAVE_DIR, filenames[current_stage_idx])
                        cv2.imwrite(save_path, frame)
                        print(f"Captured {target_view}!")
                        
                        # Flash effect
                        cv2.rectangle(frame, (0,0), (w,h), (255,255,255), -1)
                        cv2.imshow('Auto Capture', frame)
                        cv2.waitKey(50)
                        
                        current_stage_idx += 1
                        pose_start_time = None
                    else:
                        status_msg = f"Hold Still: {countdown:.1f}s"
                        color = (0, 255, 0) # Green
                        
                else:
                    # Reset timer if user moves or turns wrong way
                    pose_start_time = None
                    if not is_visible:
                        status_msg = "Make sure body is in frame"
                        color = (0, 0, 255)
                    elif current_orientation != target_view:
                        status_msg = f"Please turn to {target_view}"
                        color = (0, 165, 255) # Orange

                # Draw skeleton for feedback
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        else:
            status_msg = "All views captured! Press Q."
            color = (255, 215, 0) # Gold

        # --- UI OVERLAY ---
        # Black banner at top
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        
        # Status Text
        step_text = f"Step {current_stage_idx + 1}/{3}: {stages[min(current_stage_idx, 2)]}"
        if current_stage_idx >= 3: step_text = "Done"
        
        cv2.putText(frame, step_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, status_msg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Auto Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()