import unittest
import numpy as np
from comms.comms import CommsClient
import time
from PIL import Image
import os

def load_symbol_images(image_dir):
    images = []
    labels = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith('.png'):
            symbol = os.path.splitext(fname)[0]
            img_path = os.path.join(image_dir, fname)
            img = Image.open(img_path).convert('L').resize((28, 28))
            arr = np.array(img).astype(np.float32) / 255.0
            images.append(arr.flatten())
            labels.append(symbol)
    return images, labels

HOST = '127.0.0.1'
PORTS = {
    'vision': 5001,
    'hand': 5002,
    'ear': 5003,
    'mouth': 5004,
}

class TestModulesIntegration(unittest.TestCase):
    def setUp(self):
        # Wait a moment to ensure all modules are running
        time.sleep(1)

    def test_vision_module(self):
        client = CommsClient(HOST, PORTS['vision'])
        image_dir = os.path.join('data', 'images')
        images, labels = load_symbol_images(image_dir)
        for idx, (img, label) in enumerate(zip(images, labels)):
            # Test recognize
            resp = client.send({'cmd': 'recognize', 'data': img})
            self.assertIsInstance(resp, dict)
            self.assertIn('label', resp)
            # Test learn
            resp = client.send({'cmd': 'learn', 'data': img, 'target_idx': idx})
            self.assertIsInstance(resp, dict)
            self.assertIn('error', resp)

    def test_hand_module(self):
        client = CommsClient(HOST, PORTS['hand'])
        # Test draw
        resp = client.send({'cmd': 'draw', 'symbol_idx': 0})
        self.assertIsInstance(resp, dict)
        self.assertIn('img', resp)
        # Test learn
        dummy_img = np.zeros((28,28), dtype=np.float32)
        resp = client.send({'cmd': 'learn', 'symbol_idx': 0, 'feedback_img': dummy_img})
        self.assertIsInstance(resp, dict)
        self.assertIn('error', resp)

    def test_ear_module(self):
        client = CommsClient(HOST, PORTS['ear'])
        dummy_feat = np.zeros((13,), dtype=np.float32)
        # Test recognize
        resp = client.send({'cmd': 'recognize', 'data': dummy_feat})
        self.assertIsInstance(resp, dict)
        self.assertIn('label', resp)
        # Test learn
        resp = client.send({'cmd': 'learn', 'data': dummy_feat, 'target_idx': 0})
        self.assertIsInstance(resp, dict)
        self.assertIn('error', resp)

    def test_mouth_module(self):
        client = CommsClient(HOST, PORTS['mouth'])
        # Test speak
        resp = client.send({'cmd': 'speak', 'symbol_idx': 0})
        self.assertIsInstance(resp, dict)
        self.assertIn('wave', resp)
        # Test learn
        dummy_wave = np.zeros((8000,), dtype=np.float32)
        resp = client.send({'cmd': 'learn', 'symbol_idx': 0, 'feedback_wave': dummy_wave})
        self.assertIsInstance(resp, dict)
        self.assertIn('error', resp)

if __name__ == '__main__':
    unittest.main()
