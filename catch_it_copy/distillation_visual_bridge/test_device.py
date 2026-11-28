import unittest
from unittest.mock import patch
import torch
from .runtime import validate_torch_device

class DeviceTests(unittest.TestCase):
    def test_restricted_gpu_reports_logical_range(self):
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'6'}),patch('torch.cuda.device_count',return_value=1),patch('torch.cuda.is_available',return_value=True):
            with self.assertRaisesRegex(ValueError,'valid logical devices: cuda:0'):
                validate_torch_device('cuda:6')
            self.assertEqual(validate_torch_device('cuda:0'),torch.device('cuda:0'))
    def test_no_visibility_remapping(self):
        with patch('torch.cuda.device_count',return_value=8),patch('torch.cuda.is_available',return_value=True):
            self.assertEqual(validate_torch_device('cuda:6'),torch.device('cuda:6'))
    def test_cpu_does_not_query_cuda(self):
        with patch('torch.cuda.device_count',side_effect=AssertionError):
            self.assertEqual(validate_torch_device('cpu'),torch.device('cpu'))
    def test_no_gpu(self):
        with patch('torch.cuda.device_count',return_value=0),patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):validate_torch_device('cuda:0')

if __name__=='__main__':unittest.main()
