# Bosch Intelligent Lighting System: Edge AI Paradigm Testbed

This repository contains the interactive testbed developed for the Bosch Senior Design Capstone Project. It benchmarks and visualizes the architectural differences between an End-to-End Vision-Language-Action (VLA) architecture and a Modular VLM + LLM pipeline for an intelligent, context-aware lighting system.

The application feeds real-time vision tracking into a simulated high-density 100x100 addressable LED matrix (10,000 nodes), visualizing the latency penalty of natural-language reasoning versus direct action heads.

---

## Technical Concept & Architecture

1. End-to-End VLA Pipeline
- Target Model: OpenVLA / Simlingo (~2B-7B Parameters)
- Target Hardware: NVIDIA Jetson AGX Orin (Edge Deployment)
- Mechanics: Images are processed through a vision encoder directly into unified action projection heads. Detected skeletal points and gesture classes immediately map into an action vector ([Zone, Intensity, R, G, B]) with negligible inference delay (~12 ms).

2. Modular VLM + LLM Pipeline
- Target Models: Visual Backbone (Qwen-VL / LLaVA-1.5) + Language Backbone (Llama-3 / Phi-3)
- Target Hardware: Distributed Edge or Host Processor
- Mechanics: The vision backbone generates symbolic text tokens representing scene state. A downstream LLM processes context, assesses hazard conditions, and outputs a formatted command payload. This testbed simulates this token-generation bottleneck using an asynchronous queue buffer calibrated to ~350 ms.

3. 100x100 Addressable LED Simulation
- Projects real-world 3D coordinates onto a discrete 10,000-diode grid.
- Nearest-neighbor rasterization preserves individual LED pixel boundaries to simulate physical hardware diffusion without UI rendering performance drops.

---

## Hardware Target Specifications

- Processing Unit: NVIDIA Jetson AGX Orin Developer Kit (3.3V logic)
- Lighting Array: High-density WS2812B Addressable LED Matrix (5V logic, dedicated 10A power supply)
- Interface: Bi-directional 3.3V-to-5V Logic Level Converter with heat-sink mounting

---

## Setup & Running the Prototype

### Prerequisites
- Python 3.10+
- Local Webcam

### Installation
1. Clone this private repository:
   git clone https://github.com/brantcolquitt/bosch-intelligent-lighting-testbed.git
   cd bosch-intelligent-lighting-testbed

2. Install dependencies:
   pip install opencv-python mediapipe Pillow numpy

3. Run the application:
   python lighting_ai_demo.py

(Note: On first startup, the script automatically downloads the official hand_landmarker.task asset (~17 MB) for MediaPipe 1.0+.)

---

## Pose & Lighting Action Mapping

- Open Palm: High-heat warning / Area flood | Maximum illumination strobe | Pure White (#FFFFFF)
- Closed Fist: Unoccupied / Idle room state | Low-power ambient backlight | Warm Amber (#FF8C00)
- Index Point: User workspace targeting | Focused spotlight tracking | Bright Cyan (#00FFFF)
- Peace Sign: Zone split requirement | Dual-sector illumination | Green (#00FF78)
- Spider-Man Pose: Multi-target emergency | Directional hazard beam | Magenta (#FF00B4)
- Dual Hands: Multi-occupant tracking | Perimeter & split arrays | Electric Blue (#FFC800)
"@ | Out-File -FilePath README.md -Encoding utf8
