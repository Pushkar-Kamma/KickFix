# KickFix
KickFix is an app that corrects TKD kicks in real time using technologies like Media Pipe Pose and Open CV
# Kick Fixer: AI Taekwondo Coach Project Documentation

## 1. MVP (Minimum Viable Product)
The goal is a real-time kick fixer for martial arts, primarily Taekwondo. It uses computer vision to identify technical errors and provide immediate feedback.

### Core Objectives
* **Feedback**: Provide real-time corrections based on stance, power, and speed.
* **Platform**: Mobile App (built with React Native).
* **AI Engine**: MediaPipe Pose for detection and Python for Machine Learning logic.

---

## 2. Feature Breakdown

### Basic Features (Phase 1)
* **Kick Counting**: 
    * Categorized by kick type.
    * Checks keypoint height (ensure leg goes above waist) and foot angles.
* **Kick Speed**: Measures the duration from the start of the chamber to the end of the kick.
* **Random Combo Prompts**: 
    * The screen displays a random combination for the user to perform.
    * *Note: Combos do not need deep analysis in the initial version.*

### Future Ideas
* Real-time kick position/angle feedback.
* Form suggestions and stance analysis.
* Punch combo tracking.
* Gamification: Streaks, leaderboards, and timed rounds.
* **B2B Potential**: A dashboard for instructors to track all students in a class.

---

## 3. Technical Research

### Pose Detection Models
| Model | Pros | Cons |
| :--- | :--- | :--- |
| **MoveNet** | Light weight, very efficient. | Only 1 tracking point for the foot; hard to tell if foot is bent. |
| **MediaPipe Pose** | **(Selected)** 30 FPS on phones; multiple foot points; good documentation. | Needs optimization for high-speed motion. |
| **OpenPose** | Extremely accurate. | Heavy and slow; cannot run real-time on mobile. |

### Technical Issues & Workarounds
1. **FPS (Speed)**: 
    * Standard 30 FPS might miss "impact" points. 
    * **Fix**: Force camera to 60 FPS and use **One Euro Filters** (adaptive smoothing) or **Kalman Filters** (predictive math) to fill in missing data points.
2. **Depth Analysis (Z-Axis)**: 
    * 3D depth from a 2D lens is often inaccurate.
    * **Fix**: Require the user to kick at a **90-degree profile** from the camera to convert the motion into a 2D plane analysis.
3. **Detection Integrity**: 
    * Lighting and self-occlusion (limbs hiding behind the body) can cause flickering.
    * **Fix**: Use `visibility` scores to ignore low-confidence frames and provide a "setup guide" for optimal lighting.

This is a flowchart diagram of the logic that will occur

<img width="2613" height="1586" alt="mermaid-diagram-2026-01-09-215754" src="https://github.com/user-attachments/assets/7adfa849-64fd-43e2-8be3-83e4dcd5fce9" />
---

## 4. Competition Analysis
* **Hit.Ai (London)**: App-based, analyzes speed, power, count, and technique for most martial arts.
* **Athlete Analyzer (Sweden)**: Log-based data and video analysis diary; requires manual video upload.
* **AI Taekwondo Coach**: Currently appears abandoned (low downloads, no data).
* **Sensei Ai**: High cost ($20-$30/month) but provides general feedback without active Assessment tools.

