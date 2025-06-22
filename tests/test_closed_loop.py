"""
Test script: Trains and tests each core module using real image and audio data.
Logs performance metrics for each module and ensures real learning cycles.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))
from symbols import SYMBOLS

import os
try:
    import numpy as np
except ImportError:
    print("[Test] numpy is required but not installed")
    sys.exit(1)
import time
from comms.comms import CommsClient
try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError:
    Image = ImageEnhance = ImageOps = None
    print("[Test] PIL not available, image operations disabled")
try:
    import librosa
except ImportError:
    librosa = None
    print("[Test] librosa not available, audio processing disabled")
import random
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("[Test] matplotlib not available, visualization disabled")
import string

# Config
# Character set: 0-9, A-Z, a-z, and symbols
ALL_SYMBOLS = SYMBOLS
# Use the same SYMBOLS list as all modules to maintain consistent indexing.
# Individual symbols will be skipped during the loop if data is missing.

N_CYCLES = 10
MASTERY_THRESHOLD = 0.95  # e.g., 95% accuracy or high similarity

# Ports for each module
VISION_PORT = 5001
HAND_PORT = 5002
EAR_PORT = 5003
MOUTH_PORT = 5004
HOST = '127.0.0.1'

vision = CommsClient(HOST, VISION_PORT)
hand = CommsClient(HOST, HAND_PORT)
ear = CommsClient(HOST, EAR_PORT)
mouth = CommsClient(HOST, MOUTH_PORT)

def augment_image(img):
    # Randomly apply augmentation if PIL is available
    if Image is None:
        return img
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle)
    if random.random() < 0.5:
        shift = random.randint(-2, 2)
        arr = np.array(img)
        arr = np.roll(arr, shift, axis=0)  # vertical shift
        arr = np.roll(arr, shift, axis=1)  # horizontal shift
        img = Image.fromarray(arr)
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        arr = np.array(img)
        noise = np.random.normal(0, 0.05, arr.shape)
        arr = np.clip(arr + noise, 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))
    return img

for symbol_idx, symbol in enumerate(SYMBOLS):
    print(f"\n=== Training cycles for symbol: {symbol} ===")
    img_path = f'data/images/{symbol}.png'
    audio_path = f'data/audio/{symbol}.wav'
    if not os.path.exists(img_path) or not os.path.exists(audio_path):
        print(f"Missing data for symbol {symbol}, skipping.")
        continue
    if Image is None or librosa is None:
        print("Required libraries for processing are missing, skipping symbol.")
        continue
    base_img = Image.open(img_path).convert('L').resize((28,28))
    img_arr = np.array(base_img).astype(np.float32) / 255.0
    y_audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    vision_acc = []
    hand_sim = []
    ear_acc = []
    mouth_sim = []
    for cycle in range(N_CYCLES):
        print(f"Cycle {cycle+1}")
        # Augment image for this cycle
        img = augment_image(base_img)
        x_img = np.array(img).astype(np.float32) / 255.0
        # Vision: Train and test
        vision.send({'cmd': 'learn', 'data': x_img.flatten(), 'target_idx': symbol_idx})
        vision_pred = vision.send({'cmd': 'recognize', 'data': x_img.flatten()})
        if vision_pred is None:
            print("Warning: No response from VisionModule.")
            acc = 0
        else:
            acc = 1 if vision_pred['label'] == symbol else 0
        vision_acc.append(acc)
        # Hand: Train and test
        hand_resp = hand.send({'cmd': 'draw', 'symbol_idx': symbol_idx})
        if hand_resp is None or 'img' not in hand_resp:
            print("Warning: No response from HandModule.")
            sim = 0
            hand_img = x_img  # fallback to input image to avoid crash
        else:
            hand_img = hand_resp['img']
            sim = -np.mean((hand_img.flatten() - x_img.flatten())**2)
        hand_sim.append(sim)
        hand.send({'cmd': 'learn', 'symbol_idx': symbol_idx, 'feedback_img': np.array(img).astype(np.float32) / 255.0})
        # Ear: Train and test
        ear.send({'cmd': 'learn', 'data': mfcc_mean, 'target_idx': symbol_idx})
        ear_pred = ear.send({'cmd': 'recognize', 'data': mfcc_mean})
        if ear_pred is None or 'label' not in ear_pred:
            print("Warning: No response from EarModule.")
            acc_ear = 0
        else:
            acc_ear = 1 if ear_pred['label'] == symbol else 0
        ear_acc.append(acc_ear)
        # Mouth: Train and test
        mouth_resp = mouth.send({'cmd': 'speak', 'symbol_idx': symbol_idx})
        if mouth_resp is None or 'audio' not in mouth_resp:
            print("Warning: No response from MouthModule.")
            mouth_audio = np.zeros_like(y_audio)
            sim_mouth = 0
        else:
            mouth_audio = mouth_resp['audio']
            # Ear evaluates mouth's speech
            if librosa is not None:
                mfcc_mouth = librosa.feature.mfcc(y=np.array(mouth_audio), sr=16000, n_mfcc=13)
                mfcc_mouth_mean = np.mean(mfcc_mouth, axis=1)
                sim_mouth = -np.mean((mfcc_mouth_mean - mfcc_mean)**2)
            else:
                sim_mouth = 0
        mouth_sim.append(sim_mouth)
        mouth.send({'cmd': 'learn', 'symbol_idx': symbol_idx, 'feedback_audio': y_audio[:len(mouth_audio)]})
        print(f"Vision acc: {acc}, Hand sim: {sim:.4f}, Ear acc: {acc_ear}, Mouth sim: {sim_mouth:.4f}")
        # Visualization if matplotlib is available
        if plt is not None:
            plt.clf()
            plt.subplot(2,2,1)
            plt.imshow(x_img.reshape(28,28), cmap='gray')
            plt.title(f'Input {symbol}')
            plt.axis('off')
            plt.subplot(2,2,2)
            plt.imshow(hand_img.reshape(28,28), cmap='gray')
            plt.title('Hand Output')
            plt.axis('off')
            plt.subplot(2,2,3)
            plt.plot(vision_acc, label='Vision Acc')
            plt.plot(hand_sim, label='Hand Sim')
            plt.plot(ear_acc, label='Ear Acc')
            plt.plot(mouth_sim, label='Mouth Sim')
            plt.legend()
            plt.title('Learning Curves')
            plt.tight_layout()
            plt.pause(0.01)
        time.sleep(0.1)
        # Mastery check
        if len(vision_acc) >= 5 and np.mean(vision_acc[-5:]) > MASTERY_THRESHOLD and np.mean(hand_sim[-5:]) > -0.01 and np.mean(ear_acc[-5:]) > MASTERY_THRESHOLD and np.mean(mouth_sim[-5:]) > -0.01:
            print(f"Mastered {symbol}, moving to next.")
            break
    # Save metrics for this symbol
    np.savez(f'tests/metrics_{symbol}.npz', vision_acc=vision_acc, hand_sim=hand_sim, ear_acc=ear_acc, mouth_sim=mouth_sim)
    print(f"Metrics for {symbol} saved to tests/metrics_{symbol}.npz")
if plt is not None:
    plt.show()
print("All symbols processed.")
