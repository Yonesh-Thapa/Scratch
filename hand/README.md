# HandModule (Digital Hand)

Learns to write/draw any seen symbol on a digital canvas.
Uses its own learnable parameters to control digital pen/strokes.
Receives visual feedback from VisionModule comparing its drawing to the true symbol.
Learns to write by minimizing visual error (Delta Rule, Rescorla-Wagner, etc.).
Runs as an independent process, communicates via sockets/IPC.

## Overview
The HandModule learns to write/draw any seen symbol on a digital canvas using real sensorimotor feedback. It uses the Delta Rule for learning and can be extended to other biologically plausible learning rules. The module runs independently and communicates with other modules via sockets (no function calls or global state).

## Features
- Learns to draw symbols (A, B, C, 1, 2, 3) on a digital canvas (28x28 pixels)
- Receives visual feedback from VisionModule for learning
- Uses real, updatable parameters (no placeholders)
- Runs as a standalone process

## Usage
Run the module independently:
```bash
python hand_module.py --lr 0.01
```

## Communication Protocol
- Listens on TCP port 5002
- Accepts commands:
  - `draw`: Given a symbol index, returns the generated image
  - `learn`: Given a symbol index and feedback image, updates parameters using the Delta Rule

## Energy Efficiency & Minimal Data Retention
- Only stores compressed parameters (no raw images)
- All computation is optimized for minimal resource use

## Path of Least Action
- All learning and drawing is performed in the most direct, efficient way possible

---
See the main system README for architectural details.
