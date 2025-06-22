# MouthModule (Digital Mouth)

## Overview
The MouthModule synthesizes audio waveforms for each symbol using its own learnable parameters (frequency, amplitude, and waveform type). It does not use any TTS or prebuilt speech synthesis. The module runs independently and communicates with other modules via sockets (no function calls or global state).

## Features
- Synthesizes unique, self-learned digital "voice" for each symbol (A, B, C, 1, 2, 3)
- Uses learnable parameters for frequency, amplitude, and waveform type
- Receives feedback from EarModule for learning
- Runs as a standalone process

## Usage
Run the module independently:
```bash
python mouth_module.py --lr 0.01
```

## Communication Protocol
- Listens on TCP port 5004
- Accepts commands:
  - `speak`: Given a symbol index, returns the generated audio waveform
  - `learn`: Given a symbol index and feedback audio, updates parameters using the Delta Rule
  - `play`: Given an audio waveform, plays it through the system's speakers

## Energy Efficiency & Minimal Data Retention
- Only stores compressed parameters (no raw audio)
- All computation is optimized for minimal resource use

## Path of Least Action
- All learning and synthesis is performed in the most direct, efficient way possible

---
See the main system README for architectural details.
