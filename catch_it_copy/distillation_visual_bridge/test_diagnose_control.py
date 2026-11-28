import unittest
import numpy as np
from .diagnose_control import intervention

class InterventionTests(unittest.TestCase):
    def test_only_requested_channels_are_replaced(self):
        student=np.arange(20,dtype=np.float32)/20;teacher=-student
        np.testing.assert_array_equal(intervention('teacher',student,teacher,False),teacher)
        np.testing.assert_array_equal(intervention('student',student,teacher,False),student)
        np.testing.assert_array_equal(intervention('teacher-carry',student,teacher,False),teacher)
        np.testing.assert_array_equal(intervention('teacher-carry',student,teacher,True),student)
        action=intervention('teacher-base',student,teacher,False)
        np.testing.assert_array_equal(action[:2],teacher[:2]);np.testing.assert_array_equal(action[2:],student[2:])
        action=intervention('teacher-arm-hand',student,teacher,False)
        np.testing.assert_array_equal(action[:2],student[:2]);np.testing.assert_array_equal(action[2:],teacher[2:])
        np.testing.assert_array_equal(student,np.arange(20,dtype=np.float32)/20)
        with self.assertRaises(ValueError):intervention('invalid',student,teacher,False)

if __name__=='__main__':unittest.main()
