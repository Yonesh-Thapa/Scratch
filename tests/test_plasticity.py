"""
test_plasticity.py
Unit test for proportional plasticity modulation in ReconstructionTrainer.
"""
import unittest
from core.reconstruction_trainer import ReconstructionTrainer
import numpy as np

class TestPlasticity(unittest.TestCase):
    def setUp(self):
        self.trainer = ReconstructionTrainer((28,28), 10, 0.05, 0.01)

    def test_proportional_plasticity(self):
        orig_lr = self.trainer.lr
        self.trainer.set_plasticity(multiplier=1.5)
        self.assertGreater(self.trainer.lr, orig_lr)
        self.trainer.set_plasticity(multiplier=0.5)
        self.assertLess(self.trainer.lr, orig_lr*1.5)
        # Test clamping
        self.trainer.set_plasticity(multiplier=1e6)
        self.assertLessEqual(self.trainer.lr, 0.1)
        self.trainer.set_plasticity(multiplier=1e-6)
        self.assertGreaterEqual(self.trainer.lr, 1e-5)

if __name__ == "__main__":
    unittest.main()
