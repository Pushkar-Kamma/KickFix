import cv2 as cv
import mediapipe as mp
import numpy as np

# --- THE FIX: IMPORT DIRECTLY ---
# Instead of mp.solutions.pose, we import it from the exact file path
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
# --------------------------------

cap = cv.VideoCapture(0)

# Use 'mp_pose.Pose' directly (removed 'mp.solutions')
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        results = pose.process(frame_rgb)
        
        frame_rgb.flags.writeable = True
        frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
            
        cv.imshow("Frame", frame)
        if cv.waitKey(1) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()