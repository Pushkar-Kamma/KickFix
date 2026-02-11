import cv2 as cv
import mediapipe as mp
import numpy as np
import time

# --- CONSTANTS & CONFIGURATION ---
KICK_HEIGHT_THRESHOLD = 80  # Kicking foot must be 80px higher than standing foot
MIN_FRAME_COUNT = 5         # Ignore noise
VISIBILITY_THRESHOLD = 0.6  # Confidence check (Stops "Ghost Kicks")
AVG_LEG_LENGTH_METERS = 0.9 # Approx 35 inches (used for MPH calculation)

# Landmark IDs
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32 

# --- MATH HELPER FUNCTIONS ---

def calculate_angle(a, b, c):
    """Calculates angle ABC (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_landmarks_with_vis(results, w, h):
    """
    Returns dict: {id: {'pos': (x,y), 'vis': visibility_score}}
    """
    lms = {}
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            lms[id] = {
                'pos': (int(lm.x * w), int(lm.y * h)),
                'vis': lm.visibility
            }
    return lms

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_point_line_distance(point, line_start, line_end):
    """Calculates perpendicular distance from a point to a line defined by two points."""
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)
    return np.abs(np.cross(b-a, a-p)) / np.linalg.norm(b-a)

def calculate_mph(history, k_ankle, k_knee, k_hip):
    """
    Calculates Speed in MPH using leg length scaling.
    """
    if len(history) < 2: return 0.0
    start_lms = history[0]['lms']
    
    # 1. Calculate Leg Length in Pixels (Hip -> Knee -> Ankle)
    leg_len_px = calculate_distance(start_lms[k_hip]['pos'], start_lms[k_knee]['pos']) + \
                 calculate_distance(start_lms[k_knee]['pos'], start_lms[k_ankle]['pos'])
    
    if leg_len_px == 0: return 0.0

    # 2. Determine Scale (Meters per Pixel)
    meters_per_px = AVG_LEG_LENGTH_METERS / leg_len_px

    max_velocity_mph = 0.0
    
    # 3. Calculate Velocity
    for i in range(1, len(history)):
        t1, pos1 = history[i-1]['time'], history[i-1]['lms'][k_ankle]['pos']
        t2, pos2 = history[i]['time'], history[i]['lms'][k_ankle]['pos']
        
        dt = t2 - t1
        if dt == 0: continue
        
        dist_px = calculate_distance(pos1, pos2)
        velocity_px_sec = dist_px / dt
        
        # Convert to real world units
        velocity_m_s = velocity_px_sec * meters_per_px
        velocity_mph = velocity_m_s * 2.23694 # Convert m/s to mph
        
        if velocity_mph > max_velocity_mph:
            max_velocity_mph = velocity_mph

    return round(max_velocity_mph, 1)

# --- ANALYSIS ENGINES ---

def check_guard(lm, k_wrist, s_wrist, shoulders_y):
    """Returns True if at least one wrist is above shoulder level."""
    # Note: smaller Y means higher on screen
    return (lm[k_wrist]['pos'][1] < shoulders_y) or (lm[s_wrist]['pos'][1] < shoulders_y)

def check_height(lm, k_ankle, k_hip):
    """Returns True if Ankle is higher (smaller Y) than Hip."""
    return lm[k_ankle]['pos'][1] < lm[k_hip]['pos'][1]

def analyze_roundhouse(history, active_leg):
    if active_leg == "Left":
        k_hip, k_knee, k_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        s_hip, s_knee, s_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        k_wrist, s_wrist = LEFT_WRIST, RIGHT_WRIST
        k_toe = LEFT_FOOT_INDEX
    else:
        k_hip, k_knee, k_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        s_hip, s_knee, s_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        k_wrist, s_wrist = RIGHT_WRIST, LEFT_WRIST
        k_toe = RIGHT_FOOT_INDEX

    best_frame_idx = 0
    max_angle = 0
    
    # Find Peak
    for i, frame in enumerate(history):
        lm = frame['lms']
        angle = calculate_angle(lm[k_hip]['pos'], lm[k_knee]['pos'], lm[k_ankle]['pos'])
        if angle > max_angle:
            max_angle = angle
            best_frame_idx = i

    best_frame = history[best_frame_idx]
    lm = best_frame['lms']
    speed_mph = calculate_mph(history, k_ankle, k_knee, k_hip)
    feedback = []
    scorecard = {"Type": "Roundhouse", "Leg": active_leg, "Errors": []}
    
    feedback.append(f"Speed: {speed_mph} MPH")

    # 1. Guard Check
    shoulders_y = max(lm[LEFT_SHOULDER]['pos'][1], lm[RIGHT_SHOULDER]['pos'][1])
    if not check_guard(lm, k_wrist, s_wrist, shoulders_y):
        feedback.append("Hands Dropped!")
        scorecard["Errors"].append("Dropped Guard")

    # 2. Height Check
    if not check_height(lm, k_ankle, k_hip):
        feedback.append("Kick Higher! (Below Belt)")
        scorecard["Errors"].append("Low Kick")

    # 3. Extension
    if max_angle < 160:
        feedback.append(f"Bad Extension: {int(max_angle)}°")
        scorecard["Errors"].append("Poor Extension")
    else:
        feedback.append(f"Great Snap! {int(max_angle)}°")

    # 4. Hip Turnover (Y-Axis)
    if lm[k_hip]['pos'][1] > lm[s_hip]['pos'][1] + 30: 
        feedback.append("Turn Hips Over!")
        scorecard["Errors"].append("No Hip Turnover")

    # 5. Foot Direction (Shin Horizontal)
    shin_diff = abs(lm[k_ankle]['pos'][1] - lm[k_knee]['pos'][1])
    if shin_diff > 100:
        feedback.append("Level your shin!")
        scorecard["Errors"].append("Shin not horizontal")
        
    # 6. Standing Leg Stability
    standing_angle = calculate_angle(lm[s_hip]['pos'], lm[s_knee]['pos'], lm[s_ankle]['pos'])
    if standing_angle < 135:
        feedback.append("Stand Tall! (Knee collapsing)")
        scorecard["Errors"].append("Standing Leg Too Bent")

    # 7. Recoil (Advanced)
    # Check if angle decreases after peak
    if best_frame_idx < len(history) - 2:
        last_frame_angle = calculate_angle(history[-1]['lms'][k_hip]['pos'], history[-1]['lms'][k_knee]['pos'], history[-1]['lms'][k_ankle]['pos'])
        if (max_angle - last_frame_angle) < 20: # Knee didn't bend back at least 20 degrees
             feedback.append("Snap back! Don't drop leg.")
             scorecard["Errors"].append("No Recoil (Leg Dropped)")

    # 8. Trajectory/Knee Lead (Advanced)
    # During chamber (approx frame 3 to peak/2), Knee X should be 'ahead' of Ankle X
    # 'Ahead' depends on direction. Assuming Kick moves away from standing leg.
    if best_frame_idx > 3:
        chamber_frame = history[int(best_frame_idx/2)]
        c_lm = chamber_frame['lms']
        # Distance from Standing Hip X
        knee_dist = abs(c_lm[k_knee]['pos'][0] - c_lm[s_hip]['pos'][0])
        ankle_dist = abs(c_lm[k_ankle]['pos'][0] - c_lm[s_hip]['pos'][0])
        if ankle_dist > knee_dist + 20: # Ankle is further out than knee during chamber
            feedback.append("Knee must lead! (Soccer kick)")
            scorecard["Errors"].append("Bad Trajectory (Soccer Kick)")

    print_scorecard(scorecard, speed_mph)
    return feedback

def analyze_side_kick(history, active_leg):
    if active_leg == "Left":
        k_hip, k_knee, k_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        k_heel, k_toe = LEFT_HEEL, LEFT_FOOT_INDEX
        s_knee = RIGHT_KNEE
        s_heel = RIGHT_HEEL # For power line
        k_wrist, s_wrist = LEFT_WRIST, RIGHT_WRIST
        l_shoulder = LEFT_SHOULDER 
    else:
        k_hip, k_knee, k_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        k_heel, k_toe = RIGHT_HEEL, RIGHT_FOOT_INDEX
        s_knee = LEFT_KNEE
        s_heel = LEFT_HEEL
        k_wrist, s_wrist = RIGHT_WRIST, LEFT_WRIST
        l_shoulder = RIGHT_SHOULDER

    best_frame_idx = 0
    max_angle = 0
    min_chamber_x_diff = 1000 

    for i, frame in enumerate(history):
        lm = frame['lms']
        angle = calculate_angle(lm[k_hip]['pos'], lm[k_knee]['pos'], lm[k_ankle]['pos'])
        if angle > max_angle:
            max_angle = angle
            best_frame_idx = i
        
        # Chamber Check
        if angle < 120:
            dist = abs(lm[k_knee]['pos'][0] - lm[s_knee]['pos'][0])
            if dist < min_chamber_x_diff:
                min_chamber_x_diff = dist

    best_frame = history[best_frame_idx]
    lm = best_frame['lms']
    speed_mph = calculate_mph(history, k_ankle, k_knee, k_hip)
    feedback = []
    scorecard = {"Type": "Side Kick", "Leg": active_leg, "Errors": []}
    
    feedback.append(f"Speed: {speed_mph} MPH")

    # 1. Guard Check
    shoulders_y = max(lm[LEFT_SHOULDER]['pos'][1], lm[RIGHT_SHOULDER]['pos'][1])
    if not check_guard(lm, k_wrist, s_wrist, shoulders_y):
        feedback.append("Hands Dropped!")
        scorecard["Errors"].append("Dropped Guard")

    # 2. Height Check
    if not check_height(lm, k_ankle, k_hip):
        feedback.append("Kick Higher! (Below Belt)")
        scorecard["Errors"].append("Low Kick")

    # 3. Extension
    if max_angle < 170:
        feedback.append(f"Push Harder! Only {int(max_angle)}°")
        scorecard["Errors"].append("Poor Extension")
    else:
        feedback.append(f"Solid Lockout: {int(max_angle)}°")

    # 4. Foot Blade
    if lm[k_heel]['pos'][1] > lm[k_toe]['pos'][1] + 20:
        feedback.append("Turn Toes Down (Blade)!")
        scorecard["Errors"].append("Toes Pointing Up")

    # 5. Chamber
    if min_chamber_x_diff > 120: 
        feedback.append("Deep Chamber Needed!")
        scorecard["Errors"].append("Weak Chamber")
        
    # 6. Torso Drop
    if lm[l_shoulder]['pos'][1] > lm[k_hip]['pos'][1]:
        feedback.append("Keep Chest Up! (Dropping too low)")
        scorecard["Errors"].append("Torso Dropped Below Hips")

    # 7. Power Line (Advanced)
    # Check alignment of Shoulder, Hip, Heel
    power_leak = calculate_point_line_distance(lm[k_hip]['pos'], lm[l_shoulder]['pos'], lm[k_heel]['pos'])
    if power_leak > 40: # Pixels deviation
        feedback.append("Align Joints! (Hip sticking out)")
        scorecard["Errors"].append(f"Broken Power Line (Dev: {int(power_leak)})")

    # 8. Knee Height Maintenance (Advanced)
    # Compare Knee Y at chamber (start) vs Knee Y at peak
    # Ideally knee should rise or stay level.
    if best_frame_idx > 2:
        chamber_knee_y = history[2]['lms'][k_knee]['pos'][1]
        peak_knee_y = lm[k_knee]['pos'][1]
        if peak_knee_y > chamber_knee_y + 30: # Knee dropped significantly (Y increased)
            feedback.append("Keep knee up!")
            scorecard["Errors"].append("Knee Dropped During Extension")

    print_scorecard(scorecard, speed_mph)
    return feedback

def analyze_front_snap(history, active_leg):
    if active_leg == "Left":
        k_hip, k_knee, k_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        s_hip, l_shoulder = RIGHT_HIP, LEFT_SHOULDER
        s_heel = RIGHT_HEEL
        k_wrist, s_wrist = LEFT_WRIST, RIGHT_WRIST
        k_toe, k_heel = LEFT_FOOT_INDEX, LEFT_HEEL
    else:
        k_hip, k_knee, k_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        s_hip, l_shoulder = LEFT_HIP, RIGHT_SHOULDER
        s_heel = LEFT_HEEL
        k_wrist, s_wrist = RIGHT_WRIST, LEFT_WRIST
        k_toe, k_heel = RIGHT_FOOT_INDEX, RIGHT_HEEL

    best_frame_idx = 0
    max_angle = 0
    min_fold_dist = 10000 
    
    for i, frame in enumerate(history):
        lm = frame['lms']
        angle = calculate_angle(lm[k_hip]['pos'], lm[k_knee]['pos'], lm[k_ankle]['pos'])
        if angle > max_angle:
            max_angle = angle
            best_frame_idx = i
            
        if angle < 100:
            dist = calculate_distance(lm[k_heel]['pos'], lm[k_hip]['pos'])
            if dist < min_fold_dist:
                min_fold_dist = dist
    
    best_frame = history[best_frame_idx]
    lm = best_frame['lms']
    speed_mph = calculate_mph(history, k_ankle, k_knee, k_hip)
    feedback = []
    scorecard = {"Type": "Front Snap", "Leg": active_leg, "Errors": []}
    
    feedback.append(f"Speed: {speed_mph} MPH")

    # 1. Guard Check
    shoulders_y = max(lm[LEFT_SHOULDER]['pos'][1], lm[RIGHT_SHOULDER]['pos'][1])
    if not check_guard(lm, k_wrist, s_wrist, shoulders_y):
        feedback.append("Hands Dropped!")
        scorecard["Errors"].append("Dropped Guard")

    # 2. Extension
    if max_angle < 170:
        feedback.append(f"Snap Leg! ({int(max_angle)}°)")
    else:
        feedback.append("Good Snap.")

    # 3. Height
    if lm[k_knee]['pos'][1] > lm[k_hip]['pos'][1]:
        feedback.append("Lift Knee Higher!")
        scorecard["Errors"].append("Low Knee")

    # 4. Torso Lean
    lean = abs(lm[l_shoulder]['pos'][0] - lm[s_hip]['pos'][0])
    if lean > 80: 
        feedback.append("Don't Lean Back!")
        scorecard["Errors"].append("Excessive Lean")
        
    # 5. Foot Direction
    foot_angle = calculate_angle(lm[k_knee]['pos'], lm[k_ankle]['pos'], lm[k_toe]['pos'])
    if foot_angle > 140:
        feedback.append("Pull Toes Back!")
        scorecard["Errors"].append("Toes Pointed (Danger)")
        
    # 6. Chamber Compression
    upper_leg_len = calculate_distance(lm[k_hip]['pos'], lm[k_knee]['pos'])
    if min_fold_dist > (upper_leg_len * 1.2): 
        feedback.append("Tighten your fold! (Heel to butt)")
        scorecard["Errors"].append("Loose Chamber Fold")

    print_scorecard(scorecard, speed_mph)
    return feedback

def print_scorecard(card, speed):
    print("\n" + "="*30)
    print(f"KICK REPORT: {card['Type']} ({card['Leg']})")
    print(f"SPEED: {speed} MPH")
    print("-" * 30)
    if not card['Errors']:
        print("PERFECT KICK! ✅")
    else:
        for err in card['Errors']:
            print(f"❌ {err}")
    print("="*30 + "\n")

# --- MAIN LOOP ---

def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv.VideoCapture(0)
    
    # Configure Window
    window_name = "Kick Fixer Ultimate"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1280, 720)

    # Global State
    state = "IDLE"
    kick_history = []
    active_leg = None
    feedback_display = ["Stand in frame", "Press 1: Roundhouse", "Press 2: Side Kick", "Press 3: Front Snap"]
    current_mode = "Roundhouse"
    kick_count = 0
    
    # Use higher complexity model for better visibility checks
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            h, w, _ = frame.shape
            
            key = cv.waitKey(1)
            if key == ord('q'): break
            elif key == ord('1'): current_mode = "Roundhouse"
            elif key == ord('2'): current_mode = "Side Kick"
            elif key == ord('3'): current_mode = "Front Snap"

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                # 1. VISIBILITY CHECK (The "Ghost" Fix)
                lms = get_landmarks_with_vis(results, w, h)
                
                # Check if critical joints are actually visible
                is_visible = True
                critical_points = [LEFT_ANKLE, RIGHT_ANKLE, LEFT_HIP, RIGHT_HIP]
                for p in critical_points:
                    if lms[p]['vis'] < VISIBILITY_THRESHOLD:
                        is_visible = False
                        break
                
                # Draw Skeleton (Only if visible to avoid confusion)
                if is_visible:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # --- ANKLE SEPARATION TRIGGER (The "Sitting" Fix) ---
                    
                    left_ankle_y = lms[LEFT_ANKLE]['pos'][1]
                    right_ankle_y = lms[RIGHT_ANKLE]['pos'][1]
                    
                    # Positive value means Left is HIGHER (smaller Y) than Right
                    diff = right_ankle_y - left_ankle_y
                    
                    left_kicking = False
                    right_kicking = False

                    # Hip Height Check (Prevent sitting triggers)
                    # Hips should be higher (smaller Y) than Knees
                    hips_y = (lms[LEFT_HIP]['pos'][1] + lms[RIGHT_HIP]['pos'][1]) / 2
                    knees_y = (lms[LEFT_KNEE]['pos'][1] + lms[RIGHT_KNEE]['pos'][1]) / 2
                    is_standing = hips_y < knees_y # Basic posture check

                    if is_standing:
                        if diff > KICK_HEIGHT_THRESHOLD:
                            left_kicking = True
                        elif diff < -KICK_HEIGHT_THRESHOLD:
                            right_kicking = True

                    is_kicking = left_kicking or right_kicking

                    # --- STATE MACHINE ---
                    
                    # 1. IDLE -> RECORDING
                    if state == "IDLE" and is_kicking:
                        active_leg = "Left" if left_kicking else "Right"
                        state = "RECORDING"
                        kick_history = []
                        feedback_display = ["Recording..."] 
                        print(f"Started Recording: {active_leg} ({current_mode})")

                    # 2. RECORDING
                    elif state == "RECORDING":
                        # Continue if kicking OR if we haven't lost tracking
                        if is_kicking:
                            kick_history.append({'time': time.time(), 'lms': lms})
                            cv.circle(frame, (30, 30), 15, (0, 0, 255), -1) 
                        else:
                            state = "ANALYZING"
                    
                    # 3. ANALYZING
                    elif state == "ANALYZING":
                        if len(kick_history) > MIN_FRAME_COUNT:
                            kick_count += 1
                            if current_mode == "Roundhouse":
                                feedback_display = analyze_roundhouse(kick_history, active_leg)
                            elif current_mode == "Side Kick":
                                feedback_display = analyze_side_kick(kick_history, active_leg)
                            elif current_mode == "Front Snap":
                                feedback_display = analyze_front_snap(kick_history, active_leg)
                        else:
                            print(f"Ignored noise ({len(kick_history)} frames)")
                            feedback_display = [f"Mode: {current_mode}"] 
                        
                        state = "IDLE"
                        active_leg = None
                else:
                    # Not visible
                    if state == "RECORDING": # Force stop if we lose tracking
                        state = "IDLE"
                        print("Lost tracking - Aborting kick")

            # --- UI OVERLAY ---
            cv.rectangle(frame, (0, 0), (400, 160), (50, 50, 50), -1) 
            cv.putText(frame, f"MODE: {current_mode}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(frame, f"KICKS: {kick_count}", (w - 200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            y = 60
            for line in feedback_display:
                color = (0, 255, 0) if "Great" in line or "Solid" in line else (255, 255, 255)
                if "Bad" in line or "Error" in line: color = (0, 0, 255)
                cv.putText(frame, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y += 25

            cv.imshow(window_name, frame)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()