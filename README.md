# Decentralized Modular AI System

## Overview
This system is a fully decentralized, modular AI that learns like a human child. It can see, hear, write, and speak symbols (A, B, C, 1, 2, 3) using real sensorimotor feedback loops and hybrid learning rules (Delta Rule, Rescorla-Wagner, Bayesian, etc.).

## Core Modules
- **VisionModule** (`/vision`): Recognizes symbols from images, learns visual features.
- **HandModule** (`/hand`): Learns to write/draw symbols, receives visual feedback.
- **EarModule** (`/ear`): Listens to real audio, extracts features, learns audio-symbol associations.
- **MouthModule** (`/mouth`): Synthesizes audio for each symbol using its own learnable parameters.
- **Communication Layer** (`/comms`): Handles all inter-module communication via sockets/IPC.

## Principles
- No central controller; all modules are independent.
- Real data only; no mockups or placeholders.
- Energy efficient, minimal data retention, path of least action.

## Running & Testing
Each module can be run independently. See each module's README for details.

## Directory Structure
- `/vision` - VisionModule
- `/hand` - HandModule
- `/ear` - EarModule
- `/mouth` - MouthModule
- `/comms` - Communication Layer
- `/data` - Real images/audio for training
- `/tests` - Test scripts and validation

## How to Run the System

1. **Prepare Real Data**
   - Place real symbol images (28x28 grayscale, e.g., A, B, C, 1, 2, 3) in `/data/images/`.
   - Place real audio samples (e.g., WAV files of spoken symbols) in `/data/audio/`.

2. **Start Each Module in a Separate Terminal**
   - VisionModule:
     ```bash
     python vision/vision_module.py --rule delta --lr 0.01
     ```
   - HandModule:
     ```bash
     python hand/hand_module.py --lr 0.01
     ```
   - EarModule:
     ```bash
     python ear/ear_module.py --lr 0.01
     ```
   - MouthModule:
     ```bash
     python mouth/mouth_module.py --lr 0.01
     ```

3. **Run Closed-Loop Test**
   - In a new terminal:
     ```bash
     python tests/test_closed_loop.py
     ```
   - The test script will demonstrate learning cycles for each symbol, with real visual and audio feedback.

## System Principles
- **Energy Efficiency:** All modules are optimized for minimal computation and storage.
- **Minimal Data Retention:** Only compressed parameters are stored; raw data is not retained after learning.
- **Path of Least Action:** All learning and communication is direct and efficient.

## Module Details
- See each module's README for specific usage, protocol, and learning rules.

## Extending the System
- Add or remove modules at runtime; each module is fully independent.
- Use `/comms/comms.py` for message passing between any modules.

## Training and Testing Procedure

### Data Preparation
- Place real 28x28 grayscale images for each symbol (A–Z, 0–9, etc.) in `/data/images/`, named `A.png`, `B.png`, etc.
- Place real audio recordings for each symbol in `/data/audio/`, named `A.wav`, `B.wav`, etc.

### Training Each Module
- **VisionModule:** Trained to recognize symbols from images using the Delta Rule or Rescorla-Wagner.
- **EarModule:** Trained to identify and distinguish each symbol’s sound using MFCC features.
- **HandModule:** Trained to write/draw each symbol, improving by comparing its output to the true image using visual similarity feedback.
- **MouthModule:** Trained to synthesize each symbol’s sound from scratch, improving by comparing output to real audio using the EarModule for feedback.

### Closed-Loop Training
- For each symbol, the system:
  1. VisionModule sees the symbol image.
  2. HandModule writes the symbol; VisionModule evaluates the result.
  3. EarModule hears the symbol; MouthModule tries to speak it.
  4. EarModule listens to MouthModule and compares; MouthModule adjusts.
- Multiple learning cycles are run to ensure measurable improvement in each module (decreasing error, increasing accuracy).

### Testing and Verification
- Performance metrics are logged for each module (recognition accuracy, writing similarity, audio similarity).
- Metrics are saved to `tests/metrics.npz` for reproducibility and optional plotting.
- All learning is based on real, actual data—no mock or placeholder data.

### Troubleshooting
- If any module fails to improve or crashes, debug, retrain, or tune parameters before proceeding.
- All results are fully reproducible and documented.

---
For questions or contributions, see module READMEs and code comments for full documentation.
