🚲 TwoWheeler Risk Analyzer

YOLO-based Two-Wheeler Detection and Risk Analysis System

This project analyzes traffic scenes to detect two-wheeler vehicles (motorcycles, bicycles) and evaluate potential risk situations using computer vision techniques.

The system combines YOLO-based object detection, dense optical flow, and Bayesian-based risk evaluation to provide a modular and extensible risk analysis pipeline.

✨ Features

YOLO-based object detection

Motion analysis using dense optical flow

Bayesian risk scoring based on scene dynamics

Real-time visualization of detected risks

Modular and easy-to-extend Python codebase

📂 Project Structure

Main.py – Main entry point

detectionClass.py – YOLO-based detection

DenseOpticalFlow.py – Optical flow–based motion analysis

risk_analyzer.py – Bayesian risk evaluation logic

risk_display.py – Risk visualization

utils.py – Helper functions

🚀 Usage
python Main.py
The system processes video input, detects objects, analyzes motion using optical flow, applies Bayesian risk estimation, and visualizes risk levels in real time.

<img width="1931" height="1599" alt="gi" src="https://github.com/user-attachments/assets/6b569c8a-4ece-4c6b-9e63-c3f3cf4d7953" />
<img width="881" height="593" alt="4" src="https://github.com/user-attachments/assets/e19db4ad-0b23-445a-ac4b-f06c9ba94ac3" />
<img width="1165" height="577" alt="1" src="https://github.com/user-attachments/assets/97e03de5-3a21-4aa5-9a3b-fd78f21a925c" />
<img width="737" height="491" alt="2" src="https://github.com/user-attachments/assets/4e4e81b7-ab88-4835-9b21-1e8ebe9072db" />

