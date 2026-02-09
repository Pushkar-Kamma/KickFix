from ast import Return
import cv2 as cv
import mediapipe as mp  # Standard import
import numpy as np
import time



def main():
   
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame=cap.read()
            height,width,channels = frame.shape
            if not ret:
                break
        
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame.flags.writeable=False
            results = pose.process(frame)
            frame.flags.writeable=True                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=8, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4, circle_radius=2))
           
            if results.pose_landmarks:
                landmarks={}
                for id in range(11,32):
                    # 1. Get the landmark
                    lm = results.pose_landmarks.landmark[id]
                    
                    # 2. Normalize (Math)
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    
                    # 3. Store in the dict
                    landmarks[id] = (cx, cy)
                Angle=Protractor(landmarks[23],landmarks[25],landmarks[27])

                cv.putText(frame, str(int(Angle)), 
                           (landmarks[25][0] - 20, landmarks[25][1] - 20), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
            if cv.waitKey(1)== ord('q'):
                print(results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[11].z)
                print(results.pose_landmarks.landmark[32].x, results.pose_landmarks.landmark[32].y, results.pose_landmarks.landmark[32].z)
                print(ret)
                print(landmarks)
                break
            cv.imshow("Frame", frame)
    cap.release()
    cv.destroyAllWindows()

def Protractor(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def SideKick(landmarks):
    pass
    


state = "IDLE"           # IDLE or RECORDING
kick_history = []        # The buffer to store the kick data
current_kick_type = "Roundhouse" # User selected this    

if __name__ == "__main__":
    main()

'''if not chamber and  chamber criteria:
    state= chamber
else if  not chamber criteria and  state==chamber:
    if sidekick:
        sidekick(data)
    else if front snap:
        front snap(data)
    
    reset all variables

if chamber:
    collect data for that time period.'''  
