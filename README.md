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
