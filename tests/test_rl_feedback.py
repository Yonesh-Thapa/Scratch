"""
test_rl_feedback.py
Unit tests for RLFeedback reward scaling, plasticity modulation, and log structure.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from core.rl_feedback import RLFeedback
import unittest
import numpy as np

class TestRLFeedback(unittest.TestCase):
    def setUp(self):
        self.letters = ['A', 'B']
        self.rl = RLFeedback(self.letters)

    def test_reward_scaling(self):
        # First call, no previous error
        r, *_ = self.rl.update('A', 0.5)
        self.assertEqual(r, 0.0)
        # Improvement
        r, *_ = self.rl.update('A', 0.3)
        self.assertTrue(r > 0)
        # Worsening
        r, *_ = self.rl.update('A', 0.4)
        self.assertTrue(r < 0)
        # Stagnation
        r, *_ = self.rl.update('A', 0.4)
        self.assertTrue(r <= 0)

    def test_log_structure(self):
        self.rl.update('A', 0.5)
        self.rl.update('A', 0.3)
        log = self.rl.get_log()
        self.assertEqual(len(log), 2)
        self.assertIn('letter', log[0])
        self.assertIn('error', log[0])
        self.assertIn('reward', log[0])
        self.assertIn('valence', log[0])

if __name__ == "__main__":
    unittest.main()
