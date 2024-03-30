import os
from time import time
from typing import Dict, Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from legged_gym.envs import LeggedRobot


class Toddlerbot(LeggedRobot):
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.0 * contacts, dim=1) == 1
        return 1.0 * single_contact
