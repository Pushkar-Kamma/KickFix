import cv2 as cv
import mediapipe as mp
import numpy as np
import time

# --- CONSTANTS & CONFIGURATION ---
TRIGGER_THRESHOLD = 150   # Ankle can be 150px BELOW standing knee to start (Records earlier)
MIN_FRAME_COUNT = 5       # Capture faster kicks (lowered from 8)

# Landmark IDs
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32 # Toes

# --- MATH HELPER FUNCTIONS ---

def calculate_angle(a, b, c):
    """Calculates angle ABC (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_landmarks(results, w, h):
    """Normalize landmarks to pixel coordinates."""
    lms = {}
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            lms[id] = (int(lm.x * w), int(lm.y * h))
    return lms

def calculate_distance(p1, p2):
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_kick_metrics(history, k_ankle, k_knee, k_hip):
    """
    Calculates Speed (Body-Lengths/sec) and Impact.
    Normalized by leg length to be distance-invariant.
    """
    if len(history) < 2: return 0, 0
    
    # 1. Get Leg Length (Hip to Ankle) in first frame to normalize
    start_lms = history[0]['lms']
    leg_len_px = calculate_distance(start_lms[k_hip], start_lms[k_knee]) + \
                 calculate_distance(start_lms[k_knee], start_lms[k_ankle])
    
    if leg_len_px == 0: return 0, 0

    # 2. Calculate max velocity of ankle
    max_velocity = 0
    for i in range(1, len(history)):
        t1, pos1 = history[i-1]['time'], history[i-1]['lms'][k_ankle]
        t2, pos2 = history[i]['time'], history[i]['lms'][k_ankle]
        
        dt = t2 - t1
        if dt == 0: continue
        
        dist = calculate_distance(pos1, pos2)
        velocity_px = dist / dt
        
        # Normalize: Velocity in "Legs per second"
        norm_velocity = velocity_px / leg_len_px
        if norm_velocity > max_velocity:
            max_velocity = norm_velocity

    # 3. Heuristic Power Score (0-100)
    # 8.0 legs/sec is a pro snap. 2.0 is slow.
    power_score = min(100, (max_velocity / 8.0) * 100)
    
    return round(max_velocity, 1), int(power_score)

# --- ADVANCED ANALYSIS ENGINES ---

def analyze_roundhouse(history, active_leg):
    """
    CRITERIA:
    1. Extension: > 160 deg
    2. Hip Turnover: Kicking Hip Y <= Standing Hip Y
    3. Guard: At least one hand above shoulders/chin
    4. Height: Ankle higher than Hip
    """
    # 1. Setup IDs
    if active_leg == "Left":
        k_hip, k_knee, k_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        s_hip = RIGHT_HIP
        k_wrist, s_wrist = LEFT_WRIST, RIGHT_WRIST
    else:
        k_hip, k_knee, k_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        s_hip = LEFT_HIP
        k_wrist, s_wrist = RIGHT_WRIST, LEFT_WRIST

    # 2. Find Peak Frame & Metrics
    best_frame = None
    max_angle = 0
    
    for frame in history:
        lm = frame['lms']
        angle = calculate_angle(lm[k_hip], lm[k_knee], lm[k_ankle])
        if angle > max_angle:
            max_angle = angle
            best_frame = frame

    if not best_frame: return ["Error: No peak found"]

    # Calculate Speed/Power
    speed, power = calculate_kick_metrics(history, k_ankle, k_knee, k_hip)

    lm = best_frame['lms']
    feedback = []
    scorecard = {"Type": "Roundhouse", "Leg": active_leg, "Errors": []}

    feedback.append(f"Power: {power}/100 | Speed: {speed}")

    # CRITERION 1: Extension
    if max_angle < 160:
        feedback.append(f"Bad Extension: {int(max_angle)}°")
        scorecard["Errors"].append("Poor Extension")
    else:
        feedback.append(f"Great Snap! {int(max_angle)}°")

    # CRITERION 2: Hip Turnover
    if lm[k_hip][1] > lm[s_hip][1] + 30: 
        feedback.append("Turn Hips Over!")
        scorecard["Errors"].append("No Hip Turnover")
    
    # CRITERION 3: The Guard
    shoulder_level = max(lm[LEFT_SHOULDER][1], lm[RIGHT_SHOULDER][1])
    hands_up = (lm[k_wrist][1] < shoulder_level) or (lm[s_wrist][1] < shoulder_level)
    if not hands_up:
        feedback.append("Hands Dropped!")
        scorecard["Errors"].append("Dropped Guard")

    # CRITERION 4: Height (Ankle vs Hip)
    # Ankle Y should be less (higher) than Hip Y
    if lm[k_ankle][1] > lm[k_hip][1]:
        feedback.append("Kick Higher! (Below Belt)")
        scorecard["Errors"].append("Low Kick")

    print_scorecard(scorecard, power, speed)
    return feedback

def analyze_side_kick(history, active_leg):
    """
    CRITERIA:
    1. Extension: > 170 deg
    2. Blade Foot: Heel Y < Toe Y (Relaxed buffer)
    3. Chamber Cross: Did knee cross centerline?
    """
    if active_leg == "Left":
        k_hip, k_knee, k_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        k_heel, k_toe = LEFT_HEEL, LEFT_FOOT_INDEX
        s_hip, s_knee = RIGHT_HIP, RIGHT_KNEE
    else:
        k_hip, k_knee, k_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        k_heel, k_toe = RIGHT_HEEL, RIGHT_FOOT_INDEX
        s_hip, s_knee = LEFT_HIP, LEFT_KNEE

    # Find Peak
    best_frame = None
    max_angle = 0
    min_chamber_x_diff = 1000 

    for frame in history:
        lm = frame['lms']
        angle = calculate_angle(lm[k_hip], lm[k_knee], lm[k_ankle])
        if angle > max_angle:
            max_angle = angle
            best_frame = frame
        
        if angle < 120:
            dist = abs(lm[k_knee][0] - lm[s_knee][0])
            if dist < min_chamber_x_diff:
                min_chamber_x_diff = dist

    lm = best_frame['lms']
    speed, power = calculate_kick_metrics(history, k_ankle, k_knee, k_hip)
    feedback = []
    scorecard = {"Type": "Side Kick", "Leg": active_leg, "Errors": []}
    
    feedback.append(f"Power: {power}/100 | Speed: {speed}")

    # 1. Extension
    if max_angle < 170:
        feedback.append(f"Push Harder! Only {int(max_angle)}°")
        scorecard["Errors"].append("Poor Extension")
    else:
        feedback.append(f"Solid Lockout: {int(max_angle)}°")

    # 2. Foot Blade (Heel vs Toe)
    # RELAXED: Allow heel to be 20px lower than toe and still count as "blade"
    # Ideally Heel Y < Toe Y.
    if lm[k_heel][1] > lm[k_toe][1] + 20:
        feedback.append("Turn Toes Down!")
        scorecard["Errors"].append("Toes Pointing Up")

    # 3. Chamber Cross
    if min_chamber_x_diff > 120: 
        feedback.append("Deep Chamber Needed!")
        scorecard["Errors"].append("Weak Chamber")

    print_scorecard(scorecard, power, speed)
    return feedback

def analyze_front_snap(history, active_leg):
    """
    CRITERIA:
    1. Knee Height
    2. Torso Lean
    3. Snap Speed
    """
    if active_leg == "Left":
        k_hip, k_knee, k_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        s_hip = RIGHT_HIP
        l_shoulder = LEFT_SHOULDER
    else:
        k_hip, k_knee, k_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        s_hip = LEFT_HIP
        l_shoulder = RIGHT_SHOULDER

    best_frame = None
    max_angle = 0
    for frame in history:
        lm = frame['lms']
        angle = calculate_angle(lm[k_hip], lm[k_knee], lm[k_ankle])
        if angle > max_angle:
            max_angle = angle
            best_frame = frame
    
    lm = best_frame['lms']
    speed, power = calculate_kick_metrics(history, k_ankle, k_knee, k_hip)
    feedback = []
    scorecard = {"Type": "Front Snap", "Leg": active_leg, "Errors": []}
    
    feedback.append(f"Power: {power}/100 | Speed: {speed}")

    # 1. Extension
    if max_angle < 170:
        feedback.append(f"Snap Leg! ({int(max_angle)}°)")
    else:
        feedback.append("Good Snap.")

    # 2. Knee Height
    if lm[k_knee][1] > lm[k_hip][1]:
        feedback.append("Lift Knee Higher!")
        scorecard["Errors"].append("Low Knee")

    # 3. Torso Lean
    lean = abs(lm[l_shoulder][0] - lm[s_hip][0])
    if lean > 80: 
        feedback.append("Don't Lean Back!")
        scorecard["Errors"].append("Excessive Lean")

    print_scorecard(scorecard, power, speed)
    return feedback

def print_scorecard(card, power, speed):
    print("\n" + "="*30)
    print(f"KICK REPORT: {card['Type']} ({card['Leg']})")
    print(f"POWER: {power} | SPEED: {speed}")
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
    window_name = "Kick Fixer Pro"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 1280, 720)

    # Global State
    state = "IDLE"
    kick_history = []
    active_leg = None
    feedback_display = ["Stand in frame", "Press 1: Roundhouse", "Press 2: Side Kick", "Press 3: Front Snap"]
    current_mode = "Roundhouse"
    kick_count = 0
    
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
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
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lms = get_landmarks(results, w, h)
                
                # --- AUTO-DETECT TRIGGER (Updated) ---
                left_high = False
                right_high = False
                
                if all(k in lms for k in [LEFT_ANKLE, RIGHT_KNEE, RIGHT_ANKLE, LEFT_KNEE]):
                    # TRIGGER: Ankle can be 150px BELOW knee to start recording
                    if lms[LEFT_ANKLE][1] < (lms[RIGHT_KNEE][1] + TRIGGER_THRESHOLD):
                        left_high = True
                    if lms[RIGHT_ANKLE][1] < (lms[LEFT_KNEE][1] + TRIGGER_THRESHOLD):
                        right_high = True

                is_kicking = left_high or right_high

                # --- STATE MACHINE ---
                
                # 1. IDLE -> RECORDING
                if state == "IDLE" and is_kicking:
                    if left_high: active_leg = "Left"
                    else: active_leg = "Right"
                    
                    state = "RECORDING"
                    kick_history = []
                    feedback_display = ["Recording..."] 
                    print(f"Started Recording: {active_leg} ({current_mode})")

                # 2. RECORDING
                elif state == "RECORDING":
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

            # --- UI OVERLAY ---
            cv.rectangle(frame, (0, 0), (400, 160), (50, 50, 50), -1) 
            
            # Counters
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