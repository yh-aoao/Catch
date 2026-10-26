import copy
import unittest
from model_experiments.visual_resume import validate_resume


class ResumeContractTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(iterations=5000, device='cpu', eval_every=500,
                           eval_episodes=200, seed=123, vision={'height': 160},
                           num_envs_per_task=8)
        self.metadata = dict(teacher_hashes={'throw': 'teacher'},
                             source_asset_sha256={'run.py': 'old'})
        self.saved = dict(schema='test', config=self.config, metadata=self.metadata)

    def test_evaluation_changes_allowed_and_best_reset(self):
        config = dict(self.config, iterations=10000, eval_every=2000,
                      eval_episodes=300, skip_teacher_baseline=True)
        self.assertEqual(validate_resume(self.saved, config, self.metadata, 'test'),
                         ([], True))

    def test_training_changes_rejected(self):
        for key, value in [('seed', 42), ('vision', {'height': 224}),
                           ('num_envs_per_task', 4)]:
            with self.assertRaises(ValueError):
                validate_resume(self.saved, dict(self.config, **{key: value}),
                                self.metadata, 'test', True)

    def test_source_change_requires_explicit_acceptance(self):
        metadata = copy.deepcopy(self.metadata)
        metadata['source_asset_sha256']['run.py'] = 'new'
        with self.assertRaises(ValueError):
            validate_resume(self.saved, self.config, metadata, 'test')
        self.assertEqual(validate_resume(self.saved, self.config, metadata, 'test', True),
                         (['run.py'], True))

    def test_teachers_always_checked(self):
        metadata = copy.deepcopy(self.metadata)
        metadata['teacher_hashes']['throw'] = 'different'
        with self.assertRaises(ValueError):
            validate_resume(self.saved, self.config, metadata, 'test', True)


if __name__ == '__main__':
    unittest.main()
