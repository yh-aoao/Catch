import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .benchmark import main,summarize,FIELDS

class BenchmarkTests(unittest.TestCase):
    def test_summary(self):
        rows=[dict.fromkeys(FIELDS,value) for value in (10.,20.)]
        for row in rows:row['collected_policy_steps']=2048
        summary=summarize(rows)
        self.assertEqual(summary['seconds']['mean'],15.)
        self.assertAlmostEqual(summary['policy_steps_per_second'],4096/30)
        with self.assertRaises(ValueError):summarize([])
    def test_sequential_isolated_trials_and_alternating_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'config.json';config.write_text('{}')
            output=root/'results';commands=[]
            def run(command,**kwargs):
                commands.append(command)
                path=Path(command[command.index('--output')+1]);path.mkdir()
                value=20 if path.name.startswith('single') else 10
                (path/'summary.json').write_text(json.dumps({'seconds':{'mean':value}}))
                return SimpleNamespace(returncode=0)
            argv=['benchmark','--config',str(config),'--output',str(output),'--device','cuda:4',
                  '--single-render-device','5','--multi-render-devices','5','6','7',
                  '--iterations','50','--warmup','10','--repeats','2','--num-envs','8','--rollout-steps','64']
            with patch('sys.argv',argv),patch('subprocess.run',side_effect=run),contextlib.redirect_stdout(io.StringIO()):main()
            self.assertEqual([Path(cmd[cmd.index('--output')+1]).name for cmd in commands],['single_01','multi_01','multi_02','single_02'])
            for cmd in commands:
                self.assertEqual(cmd[cmd.index('--num-envs')+1],'8')
                self.assertEqual(cmd[cmd.index('--rollout-steps')+1],'64')
            report=json.loads((output/'comparison.json').read_text())
            self.assertEqual(report['speedup'],2)
            self.assertEqual(report['time_reduction_percent'],50)

if __name__=='__main__':unittest.main()
