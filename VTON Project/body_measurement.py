import cv2
import mediapipe as mp
import math
import numpy as np

# --- 1. CONFIGURATION (SCALING CONSTANTS) ---
CM_PER_INCH = 2.54

# --- SCALING REFERENCE ---
# ⚠️ This is the ASSUMED average hip width used as the 'ruler' for scaling. 
#    Change this value if you have a more accurate average for your target group.
AVG_HIP_WIDTH_CM = 35.0         
FOCAL_LENGTH = 650.0            # Calibrated focal length of your camera

# --- TARGET DISTANCE ---
# This is the distance *assumed* to be correct for the measurements.
FINAL_SCALE_DISTANCE = 150.0  # Assumed 150 cm

# --- 2. HELPER FUNCTIONS ---

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two (x,y) points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p2[1])**2)

def get_real_measurement_in_inches(pixel_distance, distance_cm, focal_length):
    """Converts pixel distance to real-world inches using the Pinhole Camera Model."""
    if focal_length == 0: return 0
    # Formula: Real Size (cm) = (Pixel Size * Distance) / Focal Length
    cm = (pixel_distance * distance_cm) / focal_length
    return cm / CM_PER_INCH

# --- 3. MAIN LOGIC ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

print(f"Status: Waiting for user to be at {FINAL_SCALE_DISTANCE} cm distance...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    h, w, _ = frame.shape
    image = cv2.flip(frame, 1) # Flip for selfie view
    
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    shoulder_width_in, torso_length_in = 0.0, 0.0
    status_color = (0, 0, 255) # Start as Red (Not Ready)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        def get_pt(idx): return (int(lm[idx].x * w), int(lm[idx].y * h))
        
        # --- A. CHECK FOR 150 CM DISTANCE (Simplified Check) ---
        
        hip_L_pt = get_pt(23)
        hip_R_pt = get_pt(24)
        
        # Calculate the expected pixel width of the average hip at 150 cm
        # Px = (F * R) / D
        target_px_hip = (FOCAL_LENGTH * AVG_HIP_WIDTH_CM) / FINAL_SCALE_DISTANCE
        
        # Calculate the measured pixel width of the hips
        measured_px_hip = calculate_distance(hip_L_pt, hip_R_pt)
        
        # 5 cm distance buffer translated to a pixel buffer (approximate)
        # We calculate how much the hip width changes if the distance changes by 5cm
        px_tolerance = abs(target_px_hip - (FOCAL_LENGTH * AVG_HIP_WIDTH_CM) / (FINAL_SCALE_DISTANCE + 5))
        
        if abs(measured_px_hip - target_px_hip) < px_tolerance:
            # Distance is confirmed (within 150 cm +/- 5 cm range)
            status_color = (0, 255, 0) # Green (Ready)
            
            # --- B. PERFORM MEASUREMENTS (Only if distance is confirmed) ---
            
            # 1. Shoulder Width (Landmarks 11 and 12)
            shoulder_pixel_dist = calculate_distance(get_pt(11), get_pt(12))
            shoulder_width_in = get_real_measurement_in_inches(
                shoulder_pixel_dist, 
                FINAL_SCALE_DISTANCE, FOCAL_LENGTH
            )
            
            # 2. Torso Length (Midpoint of shoulders to Midpoint of hips)
            mid_shoulder = ((get_pt(11)[0] + get_pt(12)[0]) // 2, (get_pt(11)[1] + get_pt(12)[1]) // 2)
            mid_hip = ((get_pt(23)[0] + get_pt(24)[0]) // 2, (get_pt(23)[1] + get_pt(24)[1]) // 2)
            pixel_torso_length = abs(mid_hip[1] - mid_shoulder[1])
            torso_length_in = get_real_measurement_in_inches(
                pixel_torso_length, 
                FINAL_SCALE_DISTANCE, FOCAL_LENGTH
            )
            
            # Draw final measurement lines
            cv2.line(image, get_pt(11), get_pt(12), (0, 255, 0), 2)
            cv2.line(image, mid_shoulder, mid_hip, (0, 255, 0), 2)
            status_message = "Measurements Locked!"
        
        else:
            # --- C. GUIDANCE ---
            if measured_px_hip > target_px_hip:
                status_message = "MOVE BACK (Too Close)"
            else:
                status_message = "MOVE CLOSER (Too Far)"

    else:
        status_message = "Pose not detected. Step back."

    # --- D. DISPLAY RESULTS ---
    
    cv2.rectangle(image, (10, 10), (w - 20, 150), (0, 0, 0), -1)
    
    cv2.putText(image, status_message, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    if status_color == (0, 255, 0):
        cv2.putText(image, f"Shoulder: {shoulder_width_in:.1f} inches", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(image, f"Torso: {torso_length_in:.1f} inches", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    else:
        cv2.putText(image, f"Target Distance: {FINAL_SCALE_DISTANCE} cm (+/- 5 cm)", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    cv2.imshow('Body Measurement Application', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()