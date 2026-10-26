import unittest
from unittest.mock import patch
import torch
from model_experiments.visual_resume import restore_cuda_rng


class ResumeCudaTests(unittest.TestCase):
    def test_fewer_devices_remaps_training_stream(self):
        states = [torch.tensor([i], dtype=torch.uint8) for i in range(8)]
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2), patch('torch.cuda.set_rng_state') as setter:
            restore_cuda_rng(states, 'cuda:7', 'cuda:1')
        self.assertEqual([c.args[1] for c in setter.call_args_list], [0,1,1])
        self.assertEqual(setter.call_args.args[0].item(), 7)

    def test_same_layout_eval_restore(self):
        states = [torch.tensor([i], dtype=torch.uint8) for i in range(2)]
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2), patch('torch.cuda.set_rng_state') as setter:
            restore_cuda_rng(states)
        self.assertEqual(setter.call_count, 2)

    def test_cpu_no_cuda_calls(self):
        with patch('torch.cuda.is_available', return_value=False), patch('torch.cuda.set_rng_state') as setter:
            restore_cuda_rng([torch.tensor([0])], 'cuda:7', 'cpu')
        setter.assert_not_called()

    def test_more_devices_and_invalid_checkpoint(self):
        states = [torch.tensor([9], dtype=torch.uint8)]
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=4), patch('torch.cuda.set_rng_state') as setter:
            restore_cuda_rng(states, 'cuda:0', 'cuda:3')
            self.assertEqual(setter.call_args.args[1], 3)
            with self.assertRaisesRegex(ValueError, 'Checkpoint lacks'):
                restore_cuda_rng(states, 'cuda:2', 'cuda:0')
