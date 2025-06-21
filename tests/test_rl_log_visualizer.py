"""
test_rl_log_visualizer.py
Integration test for RL log visualizer script.
"""
import unittest
import os
from core import rl_log_visualizer

class TestRLLogVisualizer(unittest.TestCase):
    def test_parse_log(self):
        # Create a fake log file
        log_content = """
Epoch 1 | Letter: A | Prev: 0.5 | Error: 0.4 | Reward: 0.1 | Plasticity: 0.01 | ValenceMA: 0.05
Epoch 2 | Letter: A | Prev: 0.4 | Error: 0.3 | Reward: 0.1 | Plasticity: 0.011 | ValenceMA: 0.06
"""
        with open("rl_feedback_log.txt", "w") as f:
            f.write(log_content)
        data = rl_log_visualizer.parse_log("rl_feedback_log.txt")
        self.assertIn('A', data)
        self.assertEqual(len(data['A']), 2)
        self.assertEqual(data['A'][0]['epoch'], 1)
        self.assertAlmostEqual(data['A'][0]['error'], 0.4)
        os.remove("rl_feedback_log.txt")

if __name__ == "__main__":
    unittest.main()
