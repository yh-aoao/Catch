from distillation_visual_aux.core import *
from visual_module_experiments.core import Student as ModuleStudent, EpisodeReplay
SCHEMA = 'distillation_aux_spatial_v1'
class Student(ModuleStudent):
    variant = 'spatial'
