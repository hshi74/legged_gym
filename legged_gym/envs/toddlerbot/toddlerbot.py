import os
from time import time
from typing import Dict, Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from legged_gym.envs import LeggedRobot


class Toddlerbot(LeggedRobot):
    def _reward_dof_pos_pitch(self):
        # Add constraint on pitch angles of the leg
        return torch.sum(
            torch.square(
                self.dof_pos[:, self.dof_indices["knee"]]
                - self.dof_pos[:, self.dof_indices["hip_pitch"]]
                - self.dof_pos[:, self.dof_indices["ank_pitch"]]
            ),
            dim=1,
        )

    def _reward_dof_pos_roll(self):
        # Add constraint on roll angles of the leg
        return torch.sum(
            torch.square(
                self.dof_pos[:, self.dof_indices["hip_roll"]]
                + self.dof_pos[:, self.dof_indices["ank_roll"]]
            ),
            dim=1,
        )

    def _reward_dof_pos_upper_body(self):
        # Penalize upper body joints
        dof_upper_body_indices = torch.cat(
            [
                self.dof_indices["sho_pitch"],
                self.dof_indices["sho_roll"],
                self.dof_indices["elb"],
            ]
        )
        return torch.sum(
            torch.square(
                self.dof_pos[:, dof_upper_body_indices]
                - self.default_dof_pos[:, dof_upper_body_indices]
            ),
            dim=1,
        )
