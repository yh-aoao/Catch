"""RNG isolation and shutdown lifecycle, without requiring server GPUs."""
import unittest
import multiprocessing as mp
from unittest.mock import patch
from types import SimpleNamespace
import torch
from .run import rng_state,restore_rng
from .environments import Pool

def blocked_sender(pipe):
    try:
        pipe.send('ready')
        pipe.send(bytes(8*1024*1024))
        pipe.recv()
    except (EOFError,BrokenPipeError,OSError):pass
    finally:pipe.close()

class ResourceTests(unittest.TestCase):
    def test_rng_only_queries_selected_gpu_and_restores_new_and_legacy(self):
        sentinel=torch.tensor([1],dtype=torch.uint8)
        with patch('torch.cuda.get_rng_state',return_value=sentinel) as get,patch('torch.cuda.get_rng_state_all',side_effect=AssertionError('All GPU access forbidden')):
            state=rng_state('cuda:5');get.assert_called_once_with(torch.device('cuda:5'))
        with patch('torch.cuda.set_rng_state') as put:
            restore_rng(state,'cuda:5','cuda:2');put.assert_called_once()
            self.assertEqual(put.call_args[0][1],torch.device('cuda:2'))
        state['cuda']=[torch.tensor([i],dtype=torch.uint8) for i in range(8)]
        with patch('torch.cuda.set_rng_state') as put:
            restore_rng(state,'cuda:5','cuda:2');put.assert_called_once()
            self.assertEqual(put.call_args[0][0].item(),5)
    def test_cpu_rng_never_touches_cuda(self):
        with patch('torch.cuda.get_rng_state',side_effect=AssertionError),patch('torch.cuda.get_rng_state_all',side_effect=AssertionError):
            state=rng_state('cpu');restore_rng(state,'cpu','cpu')
    def test_real_spawned_sender_is_reaped(self):
        context=mp.get_context('spawn');parent,child=context.Pipe()
        process=context.Process(target=blocked_sender,args=(child,),daemon=True)
        process.start();child.close()
        pool=Pool.__new__(Pool);pool.pipes=[parent];pool.processes=[process]
        try:
            self.assertTrue(parent.poll(20));self.assertEqual(parent.recv(),'ready')
        finally:pool.close()
        with self.assertRaises(ValueError):process.is_alive()

    def test_pool_close_is_idempotent_and_kills_stuck_workers(self):
        events=[]
        class Pipe:
            def send(self,value):events.append('send')
            def close(self):events.append('pipe_close')
        class Process:
            alive=True
            def join(self,timeout):events.append('join')
            def is_alive(self):return self.alive
            def terminate(self):events.append('terminate')
            def kill(self):self.alive=False;events.append('kill')
            def close(self):events.append('process_close')
        pool=Pool.__new__(Pool);pool.pipes=[Pipe()];pool.processes=[Process()];pool.obs='old observation'
        pool.close();previous=list(events);pool.close()
        self.assertEqual(events,previous)
        self.assertLess(events.index('pipe_close'),events.index('join'))
        self.assertIn('kill',events);self.assertIn('process_close',events)
        self.assertEqual(pool.processes,[]);self.assertIsNone(pool.obs)

if __name__=='__main__':unittest.main()
