# import cv2
# import mediapipe as mp
# import time
# import os

# # --- CONFIG ---
# SAVE_DIR = "captured_poses"
# VISIBILITY_THRESH = 0.8 
# STABILITY_DURATION = 1.5 

# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# def check_visibility(landmarks, required_indices):
#     for idx in required_indices:
#         lm = landmarks[idx]
#         if lm.visibility < VISIBILITY_THRESH or not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
#             return False
#     return True

# def detect_orientation(landmarks):
#     l_shoulder = landmarks[11]
#     r_shoulder = landmarks[12]
#     z_diff = l_shoulder.z - r_shoulder.z
#     if z_diff > 0.15: return "RIGHT_SIDE"
#     elif z_diff < -0.15: return "LEFT_SIDE"
#     else: return "FRONT"

# cap = cv2.VideoCapture(0)
# stages = ["FRONT", "RIGHT_SIDE", "LEFT_SIDE"]
# filenames = ["front.jpg", "right_side.jpg", "left_side.jpg"]
# current_stage_idx = 0
# pose_start_time = None

# with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
        
#         frame = cv2.flip(frame, 1)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, _ = frame.shape
#         results = pose.process(rgb_frame)
        
#         status_msg = "Position yourself..."
#         color = (0, 255, 255)

#         if current_stage_idx < len(stages):
#             target_view = stages[current_stage_idx]
#             if results.pose_landmarks:
#                 lm = results.pose_landmarks.landmark
                
#                 if target_view == "FRONT": req = [11, 12, 23, 24]
#                 elif target_view == "RIGHT_SIDE": req = [12, 24]
#                 elif target_view == "LEFT_SIDE": req = [11, 23]
                
#                 is_visible = check_visibility(lm, req)
#                 orientation = detect_orientation(lm)
                
#                 if is_visible and orientation == target_view:
#                     if pose_start_time is None: pose_start_time = time.time()
#                     elapsed = time.time() - pose_start_time
#                     countdown = max(0, STABILITY_DURATION - elapsed)
                    
#                     if countdown == 0:
#                         save_path = os.path.join(SAVE_DIR, filenames[current_stage_idx])
#                         cv2.imwrite(save_path, frame)
#                         print(f"Captured {target_view}!")
#                         current_stage_idx += 1
#                         pose_start_time = None
#                         cv2.rectangle(frame, (0,0), (w,h), (255,255,255), -1)
#                         cv2.waitKey(50)
#                     else:
#                         status_msg = f"Hold Still: {countdown:.1f}s"
#                         color = (0, 255, 0)
#                 else:
#                     pose_start_time = None
#                     status_msg = f"Please turn: {target_view}"
#                     color = (0, 0, 255)
#                 mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         else:
#             status_msg = "Capture Done! Press Q."
#             color = (255, 215, 0)

#         cv2.putText(frame, status_msg, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#         cv2.imshow('Step 1: Camera', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'): break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import math
import numpy as np

# --- 1. CONFIGURATION (INPUTS & CALIBRATION CONSTANTS) ---
CM_PER_INCH = 2.54
USER_HEIGHT_CM = 160.0   # <<< Input: The user's actual height in CM
FOCAL_LENGTH = 650.0     # Your calibrated focal length (Tweak for final accuracy)
USER_DISTANCE_CM = 100.0 # Target distance for calibration

# Reference width for guidance calculation (Used to size the guide box horizontally)
AVG_SHOULDER_CM = 45.0   

# GUIDANCE SETTINGS
PIXEL_TOLERANCE = 20     # Allowable error (in pixels)

# --- 2. HELPER FUNCTIONS ---

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two (x,y) points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_real_measurement_in_inches(pixel_distance, distance_cm, focal_length):
    """Converts pixel distance to real-world inches using the Pinhole Camera Model."""
    if focal_length == 0: return 0
    cm = (pixel_distance * distance_cm) / focal_length
    return cm / CM_PER_INCH

def get_target_pixel_height(real_height_cm, distance_cm, focal_length):
    """Calculates the target pixel height for the guide box."""
    # Px = (F * R) / D
    if distance_cm == 0: return 0
    return int((focal_length * real_height_cm) / distance_cm)

def get_target_pixel_size(real_width_cm, distance_cm, focal_length):
    """Calculates the target pixel width (size) for the guide box."""
    # Px = (F * R) / D
    if distance_cm == 0: return 0
    return int((focal_length * real_width_cm) / distance_cm)


# --- 3. INITIAL CALCULATIONS (Called Once) ---
# TARGET_PX_HEIGHT is the height of the guide box
TARGET_PX_HEIGHT = get_target_pixel_height(USER_HEIGHT_CM, USER_DISTANCE_CM, FOCAL_LENGTH)

# TARGET_PX_SHOULDER is the width of the guide box (uses AVG_SHOULDER_CM for structure)
TARGET_PX_SHOULDER = get_target_pixel_size(AVG_SHOULDER_CM, USER_DISTANCE_CM, FOCAL_LENGTH)


# --- 4. MAIN LOGIC ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

print(f"Status: Guiding user to stand {USER_DISTANCE_CM}cm away.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    h, w, _ = frame.shape
    image = cv2.flip(frame, 1) 
    
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    shoulder_width_in, torso_length_in = 0.0, 0.0
    status_message, status_color = "Aim for the box.", (0, 255, 255)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        def get_pt(idx): return (int(lm[idx].x * w), int(lm[idx].y * h))
        
        # --- A. DETECT USER HEIGHT (The Ruler) ---
        nose = get_pt(0)
        ankle_mid = ((get_pt(27)[0] + get_pt(28)[0])//2, (get_pt(27)[1] + get_pt(28)[1])//2)
        pixel_user_height = calculate_distance(nose, ankle_mid) * 1.15 
        
        # --- B. DRAW REFERENCE BOX & CHECK ---
        
        # FIX: Define center_y and center_x
        center_x = w // 2
        center_y = h // 2  # <<< FIX FOR NameError: center_y
        
        box_half_height = TARGET_PX_HEIGHT // 2
        box_half_width = TARGET_PX_SHOULDER // 2

        # Draw the target box (Red)
        cv2.rectangle(image, 
                      (center_x - box_half_width, center_y - box_half_height), 
                      (center_x + box_half_width, center_y + box_half_height), 
                      (0, 0, 255), 2) # Red Box

        # --- C. GUIDANCE AND CALCULATION ---
        
        diff_px = pixel_user_height - TARGET_PX_HEIGHT
        
        if abs(diff_px) < PIXEL_TOLERANCE:
            # READY STATE: Measurements are accurate at this distance
            
            # 1. Calculate PPCM (Pixels Per CM)
            ppcm = pixel_user_height / USER_HEIGHT_CM
            
            # 2. Calculate Final Measurements in Inches
            shoulder_width_in = get_real_measurement_in_inches(calculate_distance(get_pt(11), get_pt(12)), USER_DISTANCE_CM, FOCAL_LENGTH)
            
            mid_shoulder = ((get_pt(11)[0] + get_pt(12)[0]) // 2, (get_pt(11)[1] + get_pt(12)[1]) // 2)
            mid_hip = ((get_pt(23)[0] + get_pt(24)[0]) // 2, (get_pt(23)[1] + get_pt(24)[1]) // 2)
            pixel_torso_length = abs(mid_hip[1] - mid_shoulder[1])
            torso_length_in = get_real_measurement_in_inches(pixel_torso_length, USER_DISTANCE_CM, FOCAL_LENGTH)
            
            status_message = "READY! Measurements are accurate."
            status_color = (0, 255, 0) # Green
            
            # Draw final lines (Green for ready)
            cv2.line(image, get_pt(11), get_pt(12), (0, 255, 0), 2)


        else:
            # GUIDANCE STATE
            if diff_px > 0:
                status_message = "MOVE BACK (Too Close)"
                status_color = (0, 0, 255) # Red
            else:
                status_message = "MOVE CLOSER (Too Far)"
                status_color = (255, 165, 0) # Orange


    # --- D. DISPLAY RESULTS ---
    
    cv2.rectangle(image, (10, 10), (w - 20, 160), (0, 0, 0), -1)
    
    cv2.putText(image, status_message, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    cv2.putText(image, f"Shoulder Width: {shoulder_width_in:.1f} inches", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.putText(image, f"Torso Length: {torso_length_in:.1f} inches", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    cv2.putText(image, f"Height Target: {USER_HEIGHT_CM} cm (Fill the Red Box)", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Interactive Height Calibration', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import math
# import numpy as np
# import time

# # --- 1. CONFIGURATION (Tweak for reference accuracy) ---
# CM_PER_INCH = 2.54
# USER_HEIGHT_CM = 170.0    # Input: The user's actual height in CM
# USER_DISTANCE_CM = 100.0  # Target distance (cm) for calibration
# AVG_SHOULDER_CM = 45.0    # Reference width for calibration (known real world size)

# # GUIDANCE SETTINGS
# PIXEL_TOLERANCE = 20      # Allowable error margin in pixels for the "READY" state
# CALIBRATION_FRAMES = 30   # Number of frames to average during calibration

# # --- PHONE CAMERA INTEGRATION ---
# # Confirm the correct endpoint if '/video' fails (try '/stream' or '/live')
# IP_CAMERA_URL = 'http://10.42.5.34:8080/video' 

# # --- 2. GLOBAL STATE ---
# CALIBRATED_FOCAL_LENGTH = 0.0 # Will be calculated in the calibration step
# is_calibrated = False

# # --- 3. HELPER FUNCTIONS ---

# def calculate_distance(p1, p2):
#     """Calculates Euclidean distance between two (x,y) points."""
#     return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# def get_real_measurement_in_inches(pixel_distance, distance_cm, focal_length):
#     """Converts pixel distance to real-world inches using the Pinhole Camera Model."""
#     if focal_length == 0: return 0
#     # The real measurement (W_real) is calculated by: W_real = (P_measured * D_actual) / F_calibrated
#     cm = (pixel_distance * distance_cm) / focal_length
#     return cm / CM_PER_INCH

# def get_target_pixel_size(real_width_cm, distance_cm, focal_length):
#     """Calculates the target pixel size (P_target) for a known object at a known distance."""
#     if distance_cm == 0: return 0
#     # P_target = (W_real * F) / D_known
#     return (real_width_cm * focal_length) / distance_cm

# # --- 4. CALIBRATION LOGIC ---

# def run_calibration(cap, mp_pose, pose, h_frame, w_frame):
#     """Performs a one-time focal length calibration."""
#     global CALIBRATED_FOCAL_LENGTH, is_calibrated
    
#     print(f"Starting calibration. Please stand exactly {USER_DISTANCE_CM}cm away.")
    
#     focal_length_samples = []
#     start_time = time.time()
    
#     while len(focal_length_samples) < CALIBRATION_FRAMES:
#         ret, frame = cap.read()
#         if not ret: 
#             print("Calibration stream warning: Frame lost.")
#             time.sleep(0.1)
#             continue
        
#         image = cv2.flip(frame, 1) # Mirror image
#         h, w, _ = image.shape
#         results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
#         current_status = f"CALIBRATING ({len(focal_length_samples)}/{CALIBRATION_FRAMES}): Stand {USER_DISTANCE_CM:.0f}cm away."
#         cv2.putText(image, current_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
#         if results.pose_landmarks:
#             lm = results.pose_landmarks.landmark
#             def get_pt(idx): return (int(lm[idx].x * w), int(lm[idx].y * h))
            
#             # Use points 11 (left shoulder) and 12 (right shoulder)
#             shoulder_left = get_pt(11)
#             shoulder_right = get_pt(12)
#             pixel_shoulder_width = calculate_distance(shoulder_left, shoulder_right)
            
#             if pixel_shoulder_width > 0:
#                 # F_calibrated = (P_shoulder * D_known) / W_shoulder_known
#                 calculated_f = (pixel_shoulder_width * USER_DISTANCE_CM) / AVG_SHOULDER_CM
#                 focal_length_samples.append(calculated_f)
                
#                 # Draw visual feedback
#                 cv2.line(image, shoulder_left, shoulder_right, (0, 255, 0), 2)

#         cv2.imshow('Interactive Measurement Calibration', image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     if focal_length_samples:
#         CALIBRATED_FOCAL_LENGTH = np.median(focal_length_samples)
#         is_calibrated = True
#         print(f"Calibration Complete. Focal Length set to: {CALIBRATED_FOCAL_LENGTH:.2f}")
#     else:
#         print("Calibration Failed: Could not detect shoulders.")
        
#     cv2.destroyAllWindows()


# # --- 5. MAIN LOGIC ---

# # Initialize MediaPipe and Video Capture
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# cap = cv2.VideoCapture(IP_CAMERA_URL)

# if not cap.isOpened():
#     print("Error: Could not open video stream. Check IP_CAMERA_URL.")
#     exit()

# # Get initial frame dimensions for calibration calculation
# ret, initial_frame = cap.read()
# if not ret:
#     print("Error: Failed to read initial frame.")
#     exit()
# h_frame, w_frame, _ = initial_frame.shape
# cap.release() # Release to re-open for the main loop

# # Run calibration step
# cap = cv2.VideoCapture(IP_CAMERA_URL)
# run_calibration(cap, mp_pose, pose, h_frame, w_frame)
# cap.release() # Close calibration stream

# # Re-open for main measurement loop
# cap = cv2.VideoCapture(IP_CAMERA_URL)

# # If calibration failed, fall back to a default value (may be inaccurate)
# if not is_calibrated:
#     CALIBRATED_FOCAL_LENGTH = 700.0
#     print(f"Using default FOCAL_LENGTH={CALIBRATED_FOCAL_LENGTH:.1f} due to failed calibration.")

# # Calculate the required size for the average shoulder at the target distance using the calibrated F
# TARGET_PX_SHOULDER = get_target_pixel_size(AVG_SHOULDER_CM, USER_DISTANCE_CM, CALIBRATED_FOCAL_LENGTH)
# # Calculate the target height based on the shoulder ratio (P_height = P_shoulder * W_height / W_shoulder)
# TARGET_PX_HEIGHT = TARGET_PX_SHOULDER * (USER_HEIGHT_CM / AVG_SHOULDER_CM)


# print(f"Status: Ready to measure. Guiding user to match the target box.")

# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret: 
#         print("Warning: Stream lost or phone camera not transmitting frame.")
#         time.sleep(0.1)
#         continue 
    
#     h, w, _ = frame.shape
#     image = cv2.flip(frame, 1) # Mirror image
    
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
#     shoulder_width_in, torso_length_in = 0.0, 0.0
#     status_message, status_color = f"Aim for {USER_DISTANCE_CM:.0f}cm.", (0, 255, 255)

#     # --- Draw Target Guidance Box (based on USER_HEIGHT_CM and AVG_SHOULDER_CM) ---
#     center_x = w // 2
#     center_y = h // 2 
#     box_half_height = int(TARGET_PX_HEIGHT) // 2
#     box_half_width = int(TARGET_PX_SHOULDER) // 2

#     # Draw the target box (Red/Green based on status)
#     box_color = (0, 0, 255) # Default Red
#     cv2.rectangle(image, 
#                   (center_x - box_half_width, center_y - box_half_height), 
#                   (center_x + box_half_width, center_y + box_half_height), 
#                   box_color, 2) 

#     if results.pose_landmarks:
#         lm = results.pose_landmarks.landmark
#         def get_pt(idx): return (int(lm[idx].x * w), int(lm[idx].y * h))
        
#         # Get raw shoulder width
#         pixel_shoulder_width = calculate_distance(get_pt(11), get_pt(12))

#         # --- C. GUIDANCE AND CALCULATION ---
        
#         if abs(pixel_shoulder_width - TARGET_PX_SHOULDER) < PIXEL_TOLERANCE:
#             # --- READY STATE: Measurements are accurate at this distance ---
            
#             box_color = (0, 255, 0) # Green box for READY
#             cv2.rectangle(image, 
#                   (center_x - box_half_width, center_y - box_half_height), 
#                   (center_x + box_half_width, center_y + box_half_height), 
#                   box_color, 4) 

#             # 1. Calculate Measurements
            
#             # Since we are at the target distance, we use the target distance for the pinhole model conversion
#             shoulder_width_in = get_real_measurement_in_inches(pixel_shoulder_width, USER_DISTANCE_CM, CALIBRATED_FOCAL_LENGTH)
            
#             # Calculate Torso Length (Mid-Shoulder to Mid-Hip)
#             mid_shoulder = ((get_pt(11)[0] + get_pt(12)[0]) // 2, (get_pt(11)[1] + get_pt(12)[1]) // 2)
#             mid_hip = ((get_pt(23)[0] + get_pt(24)[0]) // 2, (get_pt(23)[1] + get_pt(24)[1]) // 2)
            
#             # Draw Torso line
#             cv2.line(image, mid_shoulder, mid_hip, (255, 0, 255), 2)
            
#             pixel_torso_length = calculate_distance(mid_shoulder, mid_hip)
#             torso_length_in = get_real_measurement_in_inches(pixel_torso_length, USER_DISTANCE_CM, CALIBRATED_FOCAL_LENGTH)
            
#             status_message = "READY! Measurements are accurate."
#             status_color = (0, 255, 0) # Green
            
#             # Draw final shoulder line (Green for ready)
#             cv2.line(image, get_pt(11), get_pt(12), (0, 255, 0), 2)

#         else:
#             # GUIDANCE STATE
#             shoulder_diff = pixel_shoulder_width - TARGET_PX_SHOULDER
            
#             if shoulder_diff > PIXEL_TOLERANCE:
#                 status_message = "MOVE BACK (Too Close)"
#                 status_color = (0, 0, 255) # Red
#             elif shoulder_diff < -PIXEL_TOLERANCE:
#                 status_message = "MOVE CLOSER (Too Far)"
#                 status_color = (255, 165, 0) # Orange
#             else:
#                 # Should be caught by the ready state, but for safety
#                 status_message = "Adjust position slightly."
#                 status_color = (0, 255, 255) # Yellow


#     # --- D. DISPLAY RESULTS ---
    
#     cv2.rectangle(image, (10, 10), (w - 20, 160), (0, 0, 0), -1)
    
#     cv2.putText(image, status_message, (20, 40), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
#     cv2.putText(image, f"Shoulder Width: {shoulder_width_in:.1f} inches", (20, 90), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
#     cv2.putText(image, f"Torso Length: {torso_length_in:.1f} inches", (20, 120), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
#     cv2.putText(image, f"Focal Length (Calibrated): {CALIBRATED_FOCAL_LENGTH:.1f}", (20, 150), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     cv2.imshow('Interactive Measurement Calibration', image)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()