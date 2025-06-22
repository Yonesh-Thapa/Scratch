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
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter
except ImportError:
    Image = ImageEnhance = ImageOps = ImageFilter = None
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
import logging
import socket

# Set up logging to both terminal and test_closed_loop.log (overwrite on each run)
logger = logging.getLogger("TestClosedLoop")
logger.setLevel(logging.INFO)
log_path = os.path.join(os.path.dirname(__file__), '../test_closed_loop.log')
file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers = [file_handler, stream_handler]

# Config
# Character set: 0-9, A-Z, a-z, and symbols
ALL_SYMBOLS = SYMBOLS
SYMBOLS = [s for s in ALL_SYMBOLS if os.path.exists(f'data/images/{s}.png') and os.path.exists(f'data/audio/{s}.wav')]

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

# Check all modules are online before starting main test loop
def check_module_status():
    status = {}
    modules = [
        ("Vision", VISION_PORT),
        ("Hand", HAND_PORT),
        ("Ear", EAR_PORT),
        ("Mouth", MOUTH_PORT),
    ]
    for name, port in modules:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect((HOST, port))
            status[name] = 'Online'
        except Exception:
            status[name] = 'Offline'
        finally:
            s.close()
    return status

status = check_module_status()
for name, stat in status.items():
    print(f"[Module Check] {name}: {stat}")
if not all(v == 'Online' for v in status.values()):
    print("[ERROR] Not all modules are online. Please start all modules before running the test.")
    sys.exit(1)

def augment_image(img):
    # Randomly apply augmentation: rotation, shift, scale, brightness, contrast, noise, blur
    import random
    import numpy as np
    from PIL import ImageEnhance
    # Rotation
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle)
    # Translation
    if random.random() < 0.5:
        shift = random.randint(-2, 2)
        arr = np.array(img)
        arr = np.roll(arr, shift, axis=0)  # vertical shift
        arr = np.roll(arr, shift, axis=1)  # horizontal shift
        img = Image.fromarray(arr)
    # Scaling
    if random.random() < 0.5:
        scale = random.uniform(0.9, 1.1)
        size = int(28 * scale)
        img = img.resize((size, size), resample=Image.BILINEAR)
        # Center crop or pad
        if scale < 1.0:
            new_img = Image.new('L', (28, 28), 0)
            new_img.paste(img, ((28 - size) // 2, (28 - size) // 2))
            img = new_img
        else:
            left = (size - 28) // 2
            img = img.crop((left, left, left + 28, left + 28))
    # Brightness
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    # Contrast
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    # Noise
    if random.random() < 0.5:
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, 0.05, arr.shape)
        arr = np.clip(arr + noise, 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))
    # Blur
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    return img

try:
    for symbol_idx, symbol in enumerate(SYMBOLS):
        logger.info(f"\n=== Training cycles for symbol: {symbol} ===")
        img_path = f'data/images/{symbol}.png'
        audio_path = f'data/audio/{symbol}.wav'
        if not os.path.exists(img_path) or not os.path.exists(audio_path):
            logger.warning(f"Missing data for symbol {symbol}, skipping.")
            continue
        base_img = Image.open(img_path).convert('L').resize((28,28))
        # DEBUG: Save and log the loaded image for inspection
        base_img.save(f'tests/debug_loaded_{symbol}.png')
        logger.info(f"[DEBUG] Saved loaded image for {symbol} to tests/debug_loaded_{symbol}.png")
        img_arr = np.array(base_img).astype(np.float32) / 255.0
        logger.info(f"[DEBUG] Raw pixel values for {symbol}: min={img_arr.min()}, max={img_arr.max()}, mean={img_arr.mean()}, std={img_arr.std()}")
        y_audio, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        current_lr = 0.01  # initial learning rate
        batch_size = 8
        batch_imgs = []
        batch_imgs_norm = []
        batch_targets = []
        batch_audio = []
        batch_audio_aug = []
        batch_mfcc_aug = []
        batch_hand_imgs_norm = []
        batch_hand_targets = []
        batch_ear_targets = []
        batch_mouth_targets = []
        vision_acc = []
        hand_sim = []
        ear_acc = []
        mouth_sim = []
        mistakes_vision = []
        mistakes_hand = []
        mistakes_mouth = []
        cycle = 0
        # Set up persistent figure and axes for real-time learning curves
        if plt is not None:
            fig, axs = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(f'Learning Curves and Images for {symbol}')
            # Add module status text box
            module_status_text = fig.text(0.5, 0.98, '', ha='center', va='top', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
            vision_line, = axs[0,0].plot([], [], label='Vision Acc')
            hand_line, = axs[0,1].plot([], [], label='Hand Sim')
            ear_line, = axs[1,0].plot([], [], label='Ear Acc')
            mouth_line, = axs[1,1].plot([], [], label='Mouth Sim')
            axs[0,0].set_title('Vision Module')
            axs[0,1].set_title('Hand Module')
            axs[1,0].set_title('Ear Module')
            axs[1,1].set_title('Mouth Module')
            axs[0,0].set_ylabel('Accuracy')
            axs[1,0].set_ylabel('Accuracy')
            axs[0,1].set_ylabel('Similarity')
            axs[1,1].set_ylabel('Similarity')
            for ax in axs[:,:2].flat:
                ax.set_xlabel('Cycle')
                ax.legend()
            # Add image display axes
            axs[0,2].set_title('Input Image')
            axs[1,2].set_title('Hand Output')
            axs[1,1].set_title('Hand - Target Diff')
            axs[0,2].axis('off')
            axs[1,2].axis('off')
            input_img_disp = axs[0,2].imshow(np.zeros((28,28)), cmap='gray', vmin='0', vmax='1')
            hand_img_disp = axs[1,2].imshow(np.zeros((28,28)), cmap='gray', vmin='0', vmax='1')
            diff_img_disp = axs[1,1].imshow(np.zeros((28,28)), cmap='bwr', vmin=-1, vmax=1)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.ion()
            plt.show()
        while True:
            cycle += 1
            if cycle % 200 == 0:
                current_lr *= 0.9
                logger.info(f"[Scheduler] Learning rate reduced to {current_lr:.6f}")
            # Augment image for this cycle
            # DEBUG: Optionally disable augmentation for Vision to test learning
            # img = augment_image(base_img)
            img = base_img  # <--- Disable augmentation for Vision debugging
            x_img = np.array(img).astype(np.float32) / 255.0
            x_img_norm = (x_img - np.mean(x_img)) / (np.std(x_img) + 1e-8)
            batch_imgs.append(x_img)
            batch_imgs_norm.append(x_img_norm)
            batch_targets.append(symbol_idx)
            # Log Vision batch data and targets for debugging
            if len(batch_imgs) == batch_size:
                logger.info(f"[Vision] batch_imgs shape: {np.stack(batch_imgs).shape}, dtype: {np.stack(batch_imgs).dtype}, targets: {batch_targets}")
            # Ear/Mouth: add pitch shift and noise
            y_audio_aug = y_audio + np.random.normal(0, 0.01, y_audio.shape)
            if random.random() < 0.5:
                y_audio_aug = librosa.effects.pitch_shift(y_audio_aug, sr=sr, n_steps=random.uniform(-2, 2))
            batch_audio.append(y_audio)
            batch_audio_aug.append(y_audio_aug)
            mfcc_aug = librosa.feature.mfcc(y=y_audio_aug, sr=sr, n_mfcc=13)
            mfcc_mean_aug = np.mean(mfcc_aug, axis=1)
            batch_mfcc_aug.append(mfcc_mean_aug)
            batch_ear_targets.append(symbol_idx)
            batch_hand_imgs_norm.append(x_img_norm)
            batch_hand_targets.append(x_img_norm)
            batch_mouth_targets.append(y_audio[:len(y_audio)])
            # Only update when batch is full
            if len(batch_imgs) == batch_size:
                # Log batch_mfcc_aug shape and dtype for debugging
                logger.info(f"[Ear] batch_mfcc_aug shape: {np.stack(batch_mfcc_aug).shape}, dtype: {np.stack(batch_mfcc_aug).dtype}")
                try:
                    # Vision batch
                    vision.send({'cmd': 'learn', 'data': np.stack(batch_imgs).reshape(batch_size, -1), 'target_idx': batch_targets, 'lr': current_lr})
                    # Hand batch
                    hand.send({'cmd': 'learn', 'symbol_idx': batch_targets, 'feedback_img': np.stack(batch_hand_imgs_norm), 'lr': current_lr})
                    # Ear batch
                    ear.send({'cmd': 'learn', 'data': np.stack(batch_mfcc_aug), 'target_idx': batch_ear_targets, 'lr': current_lr})
                    # Mouth batch
                    mouth.send({'cmd': 'learn', 'symbol_idx': batch_targets, 'feedback_wave': np.stack(batch_mouth_targets), 'lr': current_lr})
                except Exception as e:
                    logger.error(f"[Batch Send] Exception during batch send: {e}")
                batch_imgs.clear()
                batch_imgs_norm.clear()
                batch_targets.clear()
                batch_audio.clear()
                batch_audio_aug.clear()
                batch_mfcc_aug.clear()
                batch_hand_imgs_norm.clear()
                batch_hand_targets.clear()
                batch_ear_targets.clear()
                batch_mouth_targets.clear()
            # Vision: Test
            vision_pred = vision.send({'cmd': 'recognize', 'data': x_img.flatten()})
            if vision_pred is None:
                logger.warning("No response from VisionModule.")
                acc = 0
            else:
                acc = 1 if vision_pred.get('result', vision_pred.get('label')) == symbol else 0
                if acc == 0:
                    mistakes_vision.append((cycle, vision_pred.get('result', vision_pred.get('label', None))))
            vision_acc.append(acc)
            # Hand: Test
            hand_resp = hand.send({'cmd': 'draw', 'symbol_idx': symbol_idx})
            max_hand_retries = 3
            hand_retry = 0
            sim = 0
            hand_img = x_img_norm
            while hand_retry < max_hand_retries:
                if hand_resp is None or 'img' not in hand_resp:
                    logger.warning("No response from HandModule.")
                    sim = 0
                    hand_img = x_img_norm
                else:
                    hand_img = hand_resp['img']
                    hand_img_norm = (hand_img - np.mean(hand_img)) / (np.std(hand_img) + 1e-8)
                    sim = -np.mean((hand_img_norm.flatten() - x_img_norm.flatten())**2)
                # --- Real-time update of hand output image ---
                if plt is not None:
                    hand_img_disp.set_data(hand_img)
                    diff_img = hand_img - x_img_norm
                    diff_img_disp.set_data(diff_img)
                    fig.canvas.draw_idle()
                    plt.pause(0.01)
                if sim > -0.01:
                    break  # Good enough
                # Closed-loop: send feedback and redraw
                logger.info(f"[Hand Closed-Loop] Similarity {sim:.4f} below threshold, sending feedback and redrawing (attempt {hand_retry+1})")
                hand.send({'cmd': 'learn', 'symbol_idx': symbol_idx, 'feedback_img': x_img_norm, 'lr': current_lr})
                hand_resp = hand.send({'cmd': 'draw', 'symbol_idx': symbol_idx})
                hand_retry += 1
            if sim <= -0.01:
                mistakes_hand.append((cycle, sim))
            hand_sim.append(sim)
            # Ear: Test
            ear_pred = ear.send({'cmd': 'recognize', 'data': mfcc_mean_aug})
            if ear_pred is None or ('result' not in ear_pred and 'label' not in ear_pred):
                logger.warning("No response from EarModule.")
                acc_ear = 0
            else:
                acc_ear = 1 if ear_pred.get('result', ear_pred.get('label')) == symbol else 0
            ear_acc.append(acc_ear)
            # Mouth: Test
            mouth_resp = mouth.send({'cmd': 'speak', 'symbol_idx': symbol_idx})
            if mouth_resp is None or 'wave' not in mouth_resp:
                logger.warning("No response from MouthModule.")
                mouth_audio = np.zeros_like(y_audio)
                sim_mouth = 0
            else:
                mouth_audio = mouth_resp['wave']
                mfcc_mouth = librosa.feature.mfcc(y=np.array(mouth_audio), sr=16000, n_mfcc=13)
                mfcc_mouth_mean = np.mean(mfcc_mouth, axis=1)
                sim_mouth = -np.mean((mfcc_mouth_mean - mfcc_mean_aug)**2)
                mistakes_mouth.append((cycle, sim_mouth))
            mouth_sim.append(sim_mouth)
            logger.info(f"Vision acc: {acc}, Hand sim: {sim:.4f}, Ear acc: {acc_ear}, Mouth sim: {sim_mouth:.4f}")
            # --- Real-time learning curve update with moving averages ---
            def moving_avg(arr, window=10):
                if len(arr) < window:
                    return [np.mean(arr[:i+1]) for i in range(len(arr))]
                return [np.mean(arr[max(0,i-window+1):i+1]) for i in range(len(arr))]
            if plt is not None:
                vision_line.set_data(range(1, len(vision_acc)+1), vision_acc)
                hand_line.set_data(range(1, len(hand_sim)+1), hand_sim)
                ear_line.set_data(range(1, len(ear_acc)+1), ear_acc)
                mouth_line.set_data(range(1, len(mouth_sim)+1), mouth_sim)
                # Moving average lines
                if 'vision_ma_line' not in locals():
                    vision_ma_line, = axs[0,0].plot([], [], 'r--', label='Vision MA')
                    hand_ma_line, = axs[0,1].plot([], [], 'r--', label='Hand MA')
                    ear_ma_line, = axs[1,0].plot([], [], 'r--', label='Ear MA')
                    mouth_ma_line, = axs[1,1].plot([], [], 'r--', label='Mouth MA')
                vision_ma_line.set_data(range(1, len(vision_acc)+1), moving_avg(vision_acc))
                hand_ma_line.set_data(range(1, len(hand_sim)+1), moving_avg(hand_sim))
                ear_ma_line.set_data(range(1, len(ear_acc)+1), moving_avg(ear_acc))
                mouth_ma_line.set_data(range(1, len(mouth_sim)+1), moving_avg(mouth_sim))
                for ax, data in zip([axs[0,0], axs[0,1], axs[1,0], axs[1,1]], [vision_acc, hand_sim, ear_acc, mouth_sim]):
                    ax.relim()
                    ax.autoscale_view()
                # Update input and hand output images
                input_img_disp.set_data(x_img)
                hand_img_to_show = hand_img if 'hand_img' in locals() else np.zeros((28,28))
                hand_img_disp.set_data(hand_img_to_show)
                # Update module status
                status = check_module_status()
                status_str = ' | '.join([f"{k}: {v}" for k, v in status.items()])
                module_status_text.set_text(f"Module Status: {status_str}")
                fig.canvas.draw_idle()
                plt.pause(0.01)
            time.sleep(0.1)
            # Mastery check
            if len(vision_acc) >= 5 and np.mean(vision_acc[-5:]) > MASTERY_THRESHOLD and np.mean(hand_sim[-5:]) > -0.01 and np.mean(ear_acc[-5:]) > MASTERY_THRESHOLD and np.mean(mouth_sim[-5:]) > -0.01:
                logger.info(f"Mastered {symbol}, moving to next.")
                break
        # Save metrics for this symbol and plot
        np.savez(f'tests/metrics_{symbol}.npz', vision_acc=vision_acc, hand_sim=hand_sim, ear_acc=ear_acc, mouth_sim=mouth_sim)
        if plt is not None:
            fig.savefig(f'tests/learning_curve_{symbol}.png')
        # Print top 3 mistakes for Vision/Hand/Mouth
        if mistakes_vision:
            logger.info(f"Top 3 Vision mistakes (cycle, predicted): {sorted(mistakes_vision, key=lambda x: x[0])[:3]}")
        if mistakes_hand:
            logger.info(f"Top 3 Hand mistakes (cycle, sim): {sorted(mistakes_hand, key=lambda x: x[1])[:3]}")
        if mistakes_mouth:
            logger.info(f"Top 3 Mouth mistakes (cycle, sim): {sorted(mistakes_mouth, key=lambda x: x[1])[:3]}")
        logger.info(f"Metrics for {symbol} saved to tests/metrics_{symbol}.npz and learning_curve_{symbol}.png")
    logger.info("All symbols processed. Exiting.")
except KeyboardInterrupt:
    logger.info("\n[Closed Loop] Interrupted by user. Shutting down modules...")
finally:
    try:
        vision.send({'cmd': 'shutdown'})
    except Exception:
        pass
    try:
        hand.send({'cmd': 'shutdown'})
    except Exception:
        pass
    try:
        ear.send({'cmd': 'shutdown'})
    except Exception:
        pass
    try:
        mouth.send({'cmd': 'shutdown'})
    except Exception:
        pass
    logger.info("[Closed Loop] All modules shutdown signal sent.")
