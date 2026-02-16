# 🔍 Deep Learning CCTV Metal Detection System

A real-time computer vision monitoring system that leverages deep learning to detect metallic objects from CCTV video streams and trigger configurable alerts.

This project combines computer vision, applied machine learning, and application security practices to simulate a practical surveillance detection pipeline.

---

## 🚀 Project Overview

The system:

* Captures live video frames from a CCTV source
* Runs inference using a trained deep learning model
* Detects metallic objects in real-time
* Triggers alert mechanisms (audio, email, webhook)
* Optionally stores screenshots upon detection
* Uses configurable sensitivity levels

The goal is to simulate how AI-powered surveillance systems can assist in automated threat detection scenarios.

---

## 🧠 Technical Stack

* Python
* Deep Learning (PyTorch-based inference pipeline)
* OpenCV for video processing
* JSON-based configuration management
* SMTP alert integration

---

## 🔐 Security Considerations

This project includes:

* Software Composition Analysis (SCA) using `pip-audit`
* Static Application Security Testing (SAST) using Bandit
* Dependency vulnerability remediation
* Secure credential handling via environment variables
* Improved exception handling for alert integrity

The security review process and findings are documented in the repository.

---

## 📌 Key Features

* Real-time inference pipeline
* Configurable detection thresholds
* Modular alerting system
* Screenshot capture on detection
* Structured logging
* Dependency hygiene and security scanning

---

## ⚠️ Disclaimer

This project is a local development prototype designed for educational and research purposes. It is not intended for production deployment without additional hardening (authentication, secure model validation, network protection, monitoring, etc.).
