import cv2 as cv
import mediapipe as mp
import numpy as np
import time

# --- CONSTANTS & CONFIGURATION ---
TRIGGER_THRESHOLD = -20   # Ankle must be 20px ABOVE standing knee to start
MIN_FRAME_COUNT = 8       # Ignore kick if fewer than 8 frames (noise)

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

# --- ADVANCED ANALYSIS ENGINES ---

def analyze_roundhouse(history, active_leg):
    """
    CRITERIA:
    1. Extension: > 160 deg
    2. Hip Turnover: Kicking Hip Y <= Standing Hip Y (Level or Higher)
    3. Guard: At least one hand above shoulders/chin
    4. Shin Alignment: Knee Y approx Equal to Ankle Y at impact
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

    # 2. Find Peak Frame (Max Extension)
    best_frame = None
    max_angle = 0
    
    for frame in history:
        lm = frame['lms']
        angle = calculate_angle(lm[k_hip], lm[k_knee], lm[k_ankle])
        if angle > max_angle:
            max_angle = angle
            best_frame = frame

    if not best_frame: return ["Error: No peak found"]

    lm = best_frame['lms']
    feedback = []
    scorecard = {"Type": "Roundhouse", "Leg": active_leg, "Errors": []}

    # CRITERION 1: Extension
    if max_angle < 160:
        feedback.append(f"Bad Extension: {int(max_angle)}° (Target 170°+)")
        scorecard["Errors"].append("Poor Extension")
    else:
        feedback.append(f"Great Snap! {int(max_angle)}°")

    # CRITERION 2: Hip Turnover
    # In pixels, Smaller Y = Higher in air. 
    # Kicking Hip (k_hip) should be higher (smaller Y) or equal to Standing Hip (s_hip)
    # We allow a 30px buffer for beginners.
    if lm[k_hip][1] > lm[s_hip][1] + 30: 
        feedback.append("Turn Hips Over! (Kicking hip too low)")
        scorecard["Errors"].append("No Hip Turnover")
    
    # CRITERION 3: The Guard (Hands)
    # Are wrists higher (smaller Y) than the lowest shoulder?
    shoulder_level = max(lm[LEFT_SHOULDER][1], lm[RIGHT_SHOULDER][1])
    hands_up = (lm[k_wrist][1] < shoulder_level) or (lm[s_wrist][1] < shoulder_level)
    
    if not hands_up:
        feedback.append("Hands Dropped! Keep guard up.")
        scorecard["Errors"].append("Dropped Guard")

    # CRITERION 4: Horizontal Shin (The 'Tabletop')
    # Knee Y and Ankle Y should be close.
    shin_slope = abs(lm[k_knee][1] - lm[k_ankle][1])
    if shin_slope > 80: # If ankle is way lower/higher than knee
        feedback.append("Level your shin (Horizontal strike)")
        scorecard["Errors"].append("Shin not horizontal")

    print_scorecard(scorecard)
    return feedback

def analyze_side_kick(history, active_leg):
    """
    CRITERIA:
    1. Extension: > 170 deg
    2. Blade Foot: Heel Y < Toe Y (Heel higher/equal)
    3. Chamber Cross: Did knee cross centerline?
    4. Linearity: Shoulder, Hip, Ankle alignment
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
    min_chamber_x_diff = 1000 # For checking cross-chamber

    for frame in history:
        lm = frame['lms']
        # Check Extension
        angle = calculate_angle(lm[k_hip], lm[k_knee], lm[k_ankle])
        if angle > max_angle:
            max_angle = angle
            best_frame = frame
        
        # Check Chamber (During lift, did Knee X cross Standing Knee X?)
        # Only check frames where leg is bent (angle < 120)
        if angle < 120:
            # Distance between Kicking Knee X and Standing Knee X
            # Ideally they should overlap or cross
            dist = abs(lm[k_knee][0] - lm[s_knee][0])
            if dist < min_chamber_x_diff:
                min_chamber_x_diff = dist

    lm = best_frame['lms']
    feedback = []
    scorecard = {"Type": "Side Kick", "Leg": active_leg, "Errors": []}

    # 1. Extension
    if max_angle < 170:
        feedback.append(f"Push Harder! Only {int(max_angle)}°")
        scorecard["Errors"].append("Poor Extension")
    else:
        feedback.append(f"Solid Lockout: {int(max_angle)}°")

    # 2. Foot Blade (Heel vs Toe)
    # Heel Y should be less (higher) than Toe Y
    if lm[k_heel][1] > lm[k_toe][1]:
        feedback.append("Turn Toes Down! Hit with Heel.")
        scorecard["Errors"].append("Toes Pointing Up")

    # 3. Chamber Cross
    if min_chamber_x_diff > 100: # Threshold for "Crossing"
        feedback.append("Deeper Chamber! Knee to chest.")
        scorecard["Errors"].append("Weak Chamber (No Cross)")

    print_scorecard(scorecard)
    return feedback

def analyze_front_snap(history, active_leg):
    """
    CRITERIA:
    1. Knee Height: Knee Y < Hip Y at extension
    2. Torso Lean: Shoulder X vs Hip X
    3. Guard: Hands up
    """
    if active_leg == "Left":
        k_hip, k_knee, k_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        s_hip = RIGHT_HIP
        l_shoulder = LEFT_SHOULDER
    else:
        k_hip, k_knee, k_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        s_hip = LEFT_HIP
        l_shoulder = RIGHT_SHOULDER # Use same side shoulder

    best_frame = None
    max_angle = 0
    for frame in history:
        lm = frame['lms']
        angle = calculate_angle(lm[k_hip], lm[k_knee], lm[k_ankle])
        if angle > max_angle:
            max_angle = angle
            best_frame = frame
    
    lm = best_frame['lms']
    feedback = []
    scorecard = {"Type": "Front Snap", "Leg": active_leg, "Errors": []}

    # 1. Extension
    if max_angle < 170:
        feedback.append(f"Snap Leg Fully! ({int(max_angle)}°)")
    else:
        feedback.append("Good Snap Extension.")

    # 2. Knee Height
    # Knee should be higher (smaller Y) than Hip
    if lm[k_knee][1] > lm[k_hip][1]:
        feedback.append("Lift Knee Higher! (Below Belt)")
        scorecard["Errors"].append("Low Knee")

    # 3. Torso Lean (Leaning Back)
    # Check X difference between Hip and Shoulder
    lean = abs(lm[l_shoulder][0] - lm[s_hip][0])
    if lean > 80: # Arbitrary pixel threshold, tune based on distance
        feedback.append("Don't Lean Back!")
        scorecard["Errors"].append("Excessive Lean")

    print_scorecard(scorecard)
    return feedback

def print_scorecard(card):
    print("\n" + "="*30)
    print(f"KICK REPORT: {card['Type']} ({card['Leg']})")
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

    # Global State
    state = "IDLE"
    kick_history = []
    active_leg = None
    feedback_display = ["Stand in frame", "Press 1: Roundhouse", "Press 2: Side Kick", "Press 3: Front Snap"]
    current_mode = "Roundhouse"
    
    # Trigger Logic Vars
    frames_since_drop = 0

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            h, w, _ = frame.shape
            
            # --- INPUT HANDLING (Mode Selection) ---
            key = cv.waitKey(1)
            if key == ord('q'): break
            elif key == ord('1'): current_mode = "Roundhouse"
            elif key == ord('2'): current_mode = "Side Kick"
            elif key == ord('3'): current_mode = "Front Snap"

            # --- PROCESS AI ---
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            
            # Draw Skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lms = get_landmarks(results, w, h)
                
                # --- AUTO-DETECT TRIGGER (Strict) ---
                # Check 1: Ankle HIGHER than Standing Knee
                left_high = False
                right_high = False
                
                # Safety check: Ensure we can see knees/ankles
                if all(k in lms for k in [LEFT_ANKLE, RIGHT_KNEE, RIGHT_ANKLE, LEFT_KNEE]):
                    
                    # Left Kick Trigger
                    if lms[LEFT_ANKLE][1] < (lms[RIGHT_KNEE][1] + TRIGGER_THRESHOLD):
                        left_high = True
                    
                    # Right Kick Trigger
                    if lms[RIGHT_ANKLE][1] < (lms[LEFT_KNEE][1] + TRIGGER_THRESHOLD):
                        right_high = True

                is_kicking = left_high or right_high

                # --- STATE MACHINE ---
                
                # 1. IDLE -> RECORDING
                if state == "IDLE" and is_kicking:
                    # Determine Leg
                    if left_high: active_leg = "Left"
                    else: active_leg = "Right"
                    
                    state = "RECORDING"
                    kick_history = []
                    feedback_display = ["Recording..."] # Clear old feedback
                    print(f"Started Recording: {active_leg} ({current_mode})")

                # 2. RECORDING
                elif state == "RECORDING":
                    if is_kicking:
                        # Append Frame
                        kick_history.append({'time': time.time(), 'lms': lms})
                        # UI Indicator
                        cv.circle(frame, (30, 30), 15, (0, 0, 255), -1) 
                    else:
                        # Kick Ended
                        state = "ANALYZING"
                
                # 3. ANALYZING
                elif state == "ANALYZING":
                    # Filter Noise (Walking detection)
                    if len(kick_history) > MIN_FRAME_COUNT:
                        if current_mode == "Roundhouse":
                            feedback_display = analyze_roundhouse(kick_history, active_leg)
                        elif current_mode == "Side Kick":
                            feedback_display = analyze_side_kick(kick_history, active_leg)
                        elif current_mode == "Front Snap":
                            feedback_display = analyze_front_snap(kick_history, active_leg)
                    else:
                        print(f"Ignored noise ({len(kick_history)} frames)")
                        feedback_display = [f"Mode: {current_mode}"] # Reset to default
                    
                    state = "IDLE"
                    active_leg = None

            # --- UI OVERLAY ---
            # Info Box
            cv.rectangle(frame, (0, 0), (400, 150), (0, 0, 0), 0) # Transparent-ish
            cv.rectangle(frame, (0, 0), (400, 150), (50, 50, 50), -1) # Background
            
            # Mode
            cv.putText(frame, f"MODE: {current_mode}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Feedback Lines
            y = 60
            for line in feedback_display:
                color = (0, 255, 0) if "Great" in line or "Solid" in line else (255, 255, 255)
                if "Bad" in line or "Error" in line: color = (0, 0, 255)
                
                cv.putText(frame, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y += 25

            cv.imshow("Kick Fixer Pro", frame)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

