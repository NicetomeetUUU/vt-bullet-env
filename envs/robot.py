import pybullet as p
import math
import numpy as np
import functools
import pybulletX as px
from collections import namedtuple

class RobotBase(object):
    """
    The base class for robots
    """

    digit_joint_names = ["left_digit_adapter_joint", "right_digit_adapter_joint"]
    def __init__(self, pos, ori):
        """
        Arguments:
            pos: [x y z]
            ori: [r p y]

        Attributes:
            id: Int, the ID of the robot
            eef_id: Int, the ID of the End-Effector
            arm_num_dofs: Int, the number of DoFs of the arm
                i.e., the IK for the EE will consider the first `arm_num_dofs` controllable (non-Fixed) joints
            joints: List, a list of joint info
            controllable_joints: List of Ints, IDs for all controllable joints
            arm_controllable_joints: List of Ints, IDs for all controllable joints on the arm (that is, the first `arm_num_dofs` of controllable joints)

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controllable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_poses: List, the rest position for all controllable joints on the arm

            gripper_range: List[Min, Max]
        """
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        pass

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]

    def __init_robot__(self):
        raise NotImplementedError
    
    def __post_load__(self):
        pass

    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def get_ee_pos(self):
        position, orientation = p.getLinkState(self.id, self.eef_id, computeForwardKinematics=True)[4: 6]
        return position, orientation

    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end', 'q_end')
        if control_method == 'q_end':
            pos, orn = self.get_ee_pos()
            new_pos = (pos[0], pos[1], pos[2] - 0.03)
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, new_pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20)
        elif control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action
        # arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

    def move_gripper(self, open_length):
        raise NotImplementedError

    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)
        
    def end_effector_index(self):
        """获取机器人末端执行器的链接ID
        Returns:
            int: 末端执行器的链接ID
        """
        return self.eef_id


class Panda(RobotBase):
    def __init_robot__(self):
        # define the robot
        # see https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_robots/panda/panda_sim_grasp.py
        self.eef_id = 11
        self.arm_num_dofs = 7
        self.arm_rest_poses = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]
        self.id = p.loadURDF('./urdf/panda.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.04]
        # create a constraint to keep the fingers centered
        c = p.createConstraint(self.id,
                               9,
                               self.id,
                               10,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    def move_gripper(self, open_length):
        assert self.gripper_range[0] <= open_length <= self.gripper_range[1]
        for i in [9, 10]:
            p.setJointMotorControl2(self.id, i, p.POSITION_CONTROL, open_length, force=20)


class UR5Robotiq85(RobotBase):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
    
    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id,
                                   self.id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)


class UR5Robotiq140(UR5Robotiq85):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        self._id = p.loadURDF('./urdf/ur5_robotiq_140.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # Robotiq 140 有更大的开合范围
        self.gripper_range = [0, 0.140]  # 0-140mm
        # if physics_client is None:
        #     physics_client = px.current_client()
        # self._physics_client = physics_client
        physics_client = px.current_client()
        self._physics_client = physics_client

    def __post_load__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': -1,
                                'left_inner_knuckle_joint': -1,
                                'right_inner_knuckle_joint': -1,
                                'left_inner_finger_joint': 1,
                                'right_inner_finger_joint': 1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)
        
    def move_gripper(self, open_length):
        """控制 Robotiq 140 夹爪的开合
        Args:
            open_length: 夹爪开合距离，单位为米，范围[0, 0.140]
        """
        # 限制开合范围
        open_length = np.clip(open_length, *self.gripper_range)
        
        # Robotiq 140 的运动学参数
        L1 = 0.140  # 指节长度
        L2 = 0.160  # 指节到夹爪尖端的距离
        
        # 计算夹爪角度
        # 当夹爪完全闭合时，角度为 0.8 弧度
        # 当夹爪完全打开时，角度为 0 弧度
        open_angle = 0.8 * (1 - open_length / self.gripper_range[1])
        
        # 控制夹爪运动
        p.setJointMotorControl2(self._id, self.mimic_parent_id, p.POSITION_CONTROL, 
                                targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, 
                                maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

    @property
    def id(self):
        return self._id
    
    @property
    def physics_client(self):
        return self._physics_client

    @property
    def _client_kwargs(self):
        return {"physicsClientId": self.physics_client.id}

    @property
    def num_joints(self):
        return p.getNumJoints(self.id, **self._client_kwargs)

    def get_joint_infos(self, joint_indices):
        """
        Get the joint informations and return JointInfo, which is a structure of arrays (SoA).
        """
        return p.getJointInfos(self.id, joint_indices, **self._client_kwargs)

    @property
    @functools.lru_cache(maxsize=None)
    def _joint_name_to_index(self):
        return {
            j.joint_name.decode(): j.joint_index
            for j in self.get_joint_infos(range(self.num_joints))
        }

    def get_joint_index_by_name(self, joint_name):
        return self._joint_name_to_index[joint_name]

    @property
    def digit_links(self):
        return [self.get_joint_index_by_name(name) for name in self.digit_joint_names]
    