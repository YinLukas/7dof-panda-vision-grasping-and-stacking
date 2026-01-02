# 7-DoF Franka Panda — Vision-Guided Grasping and Stacking

This project implements a vision-guided grasping and stacking system for a 7-DoF Franka Emika Panda robotic arm. Block poses are estimated using OpenCV, and manipulation is performed using numerical inverse kinematics with a joint-centering null-space objective. Motion planning is handled using Artificial Potential Fields (APF) and Rapidly-Exploring Random Trees (RRT).

The full pipeline was first developed and tested in the Gazebo simulation environment, where both static block grasping and dynamic grasping on a rotating stage were evaluated. The same software stack was then deployed on the real Panda arm in the lab, with calibration and parameter tuning performed to reduce sim-to-real deviations and achieve stable real-world performance.

- OpenCV-based perception  
- Numerical inverse kinematics  
- Null-space joint-centering  
- Motion planning (APF & RRT)

---

## 📷 Lab Setup

![Lab Setup](Objects%20grasping%20and%20stacking/Lab_picture.jpg)

---

## 🧱 Static Grasping & Stacking

![Static Grasping](Objects%20grasping%20and%20stacking/static_grasping.png)

---

## 🔄 Dynamic Grasping on Rotating Stage

![Dynamic Grasping](Objects%20grasping%20and%20stacking/dynamic%20grasping.png)

---

## 🏆 Final Competition Run

![Competition](Objects%20grasping%20and%20stacking/Competition.jpg)

---

More details will be added soon, including usage instructions and code structure.
