from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ToddlerbotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 66
        num_actions = 18

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.0]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": 0.325,
            "left_knee": 0.65,
            "left_ank_pitch": 0.325,
            "left_ank_roll": 0.0,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": -0.325,
            "right_knee": -0.65,
            "right_ank_pitch": -0.325,
            "right_ank_roll": 0.0,
            "left_sho_pitch": 0.0,
            "left_sho_roll": -1.57,
            "left_elb": 0.0,
            "right_sho_pitch": 0.0,
            "right_sho_roll": 1.57,
            "right_elb": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            "left_hip_yaw": 100.0,
            "left_hip_roll": 100.0,
            "left_hip_pitch": 100.0,
            "left_knee": 100.0,
            "left_ank_pitch": 100.0,
            "left_ank_roll": 100.0,
            "right_hip_yaw": 100.0,
            "right_hip_roll": 100.0,
            "right_hip_pitch": 100.0,
            "right_knee": 100.0,
            "right_ank_pitch": 100.0,
            "right_ank_roll": 100.0,
            "left_sho_pitch": 100.0,
            "left_sho_roll": 100.0,
            "left_elb": 100.0,
            "right_sho_pitch": 100.0,
            "right_sho_roll": 100.0,
            "right_elb": 100.0,
        }  # [N*m/rad]
        damping = {
            "left_hip_yaw": 4.0,
            "left_hip_roll": 4.0,
            "left_hip_pitch": 4.0,
            "left_knee": 4.0,
            "left_ank_pitch": 4.0,
            "left_ank_roll": 4.0,
            "right_hip_yaw": 4.0,
            "right_hip_roll": 4.0,
            "right_hip_pitch": 4.0,
            "right_knee": 4.0,
            "right_ank_pitch": 4.0,
            "right_ank_roll": 4.0,
            "left_sho_pitch": 4.0,
            "left_sho_roll": 4.0,
            "left_elb": 4.0,
            "right_sho_pitch": 4.0,
            "right_sho_roll": 4.0,
            "right_elb": 4.0,
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/../../robot_descriptions/toddlerbot/toddlerbot_isaac.urdf"
        name = "toddlerbot"
        foot_name = "ank_roll_link"
        terminate_after_contacts_on = [
            "body_link",
            "neck_link",
            "head_link",
            "hip_roll_link",
            "left_hip_pitch_link",
            "left_calf_link",
            "hip_roll_link_2",
            "right_hip_pitch_link",
            "right_calf_link",
            "sho_roll_link",
            "elb_link",
            "sho_roll_link_2",
            "elb_link_2",
        ]
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.0
        only_positive_rewards = False

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -200.0
            tracking_ang_vel = 1.0
            torques = -5.0e-6
            dof_acc = -2.0e-7
            lin_vel_z = -0.5
            feet_air_time = 5.0
            dof_pos_limits = -1.0
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.0
            dof_pos_pitch = -1.0
            dof_pos_roll = -1.0
            dof_pos_upper_body = -1.0

    class viewer:
        pos = [-1, -0.5, 0.5]  # [m]
        lookat = [0, 0, 0.3]  # [m]


class ToddlerbotCfgPPO(LeggedRobotCfgPPO):

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "walk_toddlerbot"
        max_iterations = 1500

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
