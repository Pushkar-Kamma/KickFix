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

---

## 4. Competition Analysis
* **Hit.Ai (London)**: App-based, analyzes speed, power, count, and technique for most martial arts.
* **Athlete Analyzer (Sweden)**: Log-based data and video analysis diary; requires manual video upload.
* **AI Taekwondo Coach**: Currently appears abandoned (low downloads, no data).
* **Sensei Ai**: High cost ($20-$30/month) but provides general feedback without active Assessment tools.

---

## 5. Logic Flowchart
You can view and edit the visual logic flow for the app's processing engine here:

[**Edit Logic Flowchart on Mermaid Live**](https://mermaid.live/edit#pako:eNptVF1v2koQ_StzV-rNC0QYcADrqhWBkC9I3JBWbU0etniwV9i71nodQlH-e8drO6VXQTLg2XPOnNmZ3QNbqxCZxyLNsxgepysJ9PnwAfyY5wiOB0s0RVaFx8HScG1gwlPUHM46MPOXT9Buf4Tzw5ccNQgJM02rn14rxjktwp2ykEkwFXmW8L0HJwv1jAQ2Cr4K3J08VehJJXVE_Y65jU0bLow67RAjjQiXscoN3D-jpnitMLXoi4OvVUZ26Gcjkjc3F0du6kr_V2_XgwfkSduIFEv2GvNcyOiI3jiaBRcvRvO1gQWGgvsiQ5hzGaZcb_PazcwiL4NxliV7uJcIF4VWMBOJIXNU_LcWfG_Bjxp-aeFXwYQn6yLhBuFG0R7BWEYJ5h5ciawFtxKxRaFtgk_vltCjlpmSPEWDayOUrABXVv36UArAJObpT9QYNltzfbQ1s6NQU-5N3Xrn1M1hyg2H82KzQf2-h75HDnmyz0VerdxYkdvgOkRpxGYPPvIt0A6izMlhNTS11q3Fzg-TGNdb8MWzMo3LeWnpP3C6HaAhsLhFME5QG5opiyxrlyG1DGZKmWawLO_jX7y7Wp82FcaJiGRKzpo81fddyVomRZqVeiXp_k-yIo9Lbk559I7rsElVkWgyRBQbS_LrTG_V_p3FLwlztbPYz28J5mJjbLMpSxSjbvQtfKK0pt5aysPRsfJRb8r4rVhv_zl5vzeuBzPE8CcnSw_4TMevAizgX7in5zM9D1Z5GSwTtWsvFOHs2dsJEzcHrpnxpYU-BgRBboBHXMh66bGZJtai-0WEzDO6wBajyyPl5Ss7lMAVMzGmuGIe_Q1xw4vErNhKvhIt4_KHUmnD1KqIYuZteJLTW5GFNOdTweny-gNBGaKeqEIa5jnOsG9FmHdgL_Q-cE97A2fYc_rdUb931hu22J557dGoezoYDZ3hwB25rtN1Xlvsl83bPXX7nf6w47qDbv-sMxz1Xn8DpoWGYA)
