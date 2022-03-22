from collections import defaultdict
import numpy as np
from lxml import etree

import metamorph.utils.camera as cu
import metamorph.utils.xml as xu
from metamorph.config import cfg
from metamorph.utils import mjpy as mu
from metamorph.utils import geom as gu


class Agent:
    def __init__(self, random_state=None):

        self.np_random = random_state

    def modify_xml_step(self, env, root, tree):
        # Store agent height
        worldbody = root.findall("./worldbody")[0]
        head = xu.find_elem(worldbody, "body", "name", "torso/0")[0]
        pos = xu.str2arr(head.get("pos"))
        orig_height = pos[2] - cfg.TERRAIN.SIZE[2]
        orig_height -= cfg.TERRAIN.FLOOR_OFFSET

        # Change position of agent to center for some tasks
        if cfg.ENV.TASK in ["point_nav", "exploration"]:
            pos[0] = cfg.TERRAIN.SIZE[0]
            head.set("pos", xu.arr2str(pos))
        if cfg.ENV.TASK == "patrol":
            pos[0] = -cfg.TERRAIN.PATROL_HALF_LEN
            head.set("pos", xu.arr2str(pos))
        if cfg.ENV.TASK in ["incline", "push_box_incline"]:
            angle = np.deg2rad(abs(cfg.TERRAIN.INCLINE_ANGLE))
            pos[0] = np.cos(angle) * (-cfg.TERRAIN.SIZE[0] + 2.0)
            height = np.sin(angle) * (cfg.TERRAIN.SIZE[0] - 2.0)
            if cfg.TERRAIN.INCLINE_ANGLE > 0:
                pos[2] += height
            else:
                pos[2] -= height
            head.set("pos", xu.arr2str(pos))
            head.set("euler", xu.arr2str([0, cfg.TERRAIN.INCLINE_ANGLE, 0]))

        # The center position in escape_bowl is ~0 height so subtract the terrain
        # height which is added in merge agent with base
        if cfg.ENV.TASK == "escape_bowl":
            pos[2] = pos[2] - cfg.TERRAIN.SIZE[2]
            head.set("pos", xu.arr2str(pos))

        self._add_cameras(head)
        self._add_fixed_cameras(worldbody)

        env.metadata["orig_height"] = round(orig_height, 2)
        env.metadata["fall_threshold"] = orig_height * cfg.ENV.STAND_HEIGHT_RATIO
        self._change_order(env, root)

    def _change_order(self, env, root):
        worldbody = root.findall("./worldbody")[0]
        root = xu.find_elem(worldbody, "body", "name", "torso/0")[0]

        def tree_treversal(order, reverse=False):
            children = xu.find_elem(order[-1], "body", child_only=True)
            if reverse:
                children = children[::-1]
            for c in children:
                order.append(c)
                tree_treversal(order, reverse=reverse)

        orig_order = [root]
        tree_treversal(orig_order)
        orig_order = [elem.get("name") for elem in orig_order]
        mirror_order = [root]
        tree_treversal(mirror_order, reverse=True)
        mirror_order = [elem.get("name") for elem in mirror_order]
        env.metadata["o_to_m"] = [
            orig_order.index(m)
            for m in mirror_order
        ]
        env.metadata["m_to_o"] = [
            mirror_order.index(o)
            for o in orig_order
        ]

    def modify_sim_step(self, env, sim):
        self.agent_qpos_idxs = np.array(mu.qpos_idxs_for_agent(sim))
        self.agent_qvel_idxs = np.array(mu.qvel_idxs_for_agent(sim))
        self.agent_geom_idxs = np.array(mu.geom_idxs_for_agent(sim))
        self.agent_body_idxs = np.array(mu.body_idxs_for_agent(sim))

        site_prefixes = ["limb/btm/", "limb/mid/", "torso"]
        env.metadata["agent_sites"] = mu.names_from_prefixes(
            sim, site_prefixes, "site"
        )

        self.limb_btm_sites = [
            site for site in env.metadata["agent_sites"] if "limb/btm" in site
        ]
        self.edges = self._get_edges(sim)
        env.metadata["num_limbs"] = len(self.agent_body_idxs)
        env.metadata["num_joints"] = len(sim.model.joint_names) - 1
        # Useful for attention map analysis
        env.metadata["edge_name"] = list(sim.model.joint_names)[1:]
        env.metadata["limb_name"] = [
            sim.model.body_names[idx] for idx in self.agent_body_idxs
        ]
        self.joint_mask_for_node_graph = self.get_joint_mask_for_node_graph(
            env.metadata["edge_name"]
        )
        env.metadata["joint_mask_for_node_graph"] = self.joint_mask_for_node_graph

    def get_joint_mask_for_node_graph(self, edge_names):
        limb_joint_types = defaultdict(list)
        limb_order = []
        for en in edge_names:
            joint_type, limb_idx = en.split("/")
            joint_type = joint_type[-1]
            limb_idx = int(limb_idx)
            if limb_idx not in limb_joint_types:
                limb_order.append(limb_idx)
            limb_joint_types[limb_idx].append(joint_type)

        # Initialize mask with values for torso/0.
        mask = [False, False]
        for limb_idx in limb_order:
            joint_types = limb_joint_types[limb_idx]
            if joint_types == ["x"]:
                mask.extend([True, False])
            elif joint_types == ["y"]:
                mask.extend([False, True])
            elif joint_types == ["x", "y"]:
                mask.extend([True, True])
        return mask

    def _get_edges(self, sim):
        body_parentids = sim.model.body_parentid.copy()
        # body idx of the child
        body_idxs = self.agent_body_idxs
        joint_to = sim.model.jnt_bodyid[1:].copy()  # ignore root
        # body idx of the parent
        joint_from = np.asarray([body_parentids[child] for child in joint_to])
        # subtract 1 from idx as idx correspond to list with first elem
        # world body
        joint_to -= 1
        joint_from -= 1
        return np.vstack((joint_to, joint_from)).T.flatten()

    def get_limb_obs(self, sim):
        obs = {}
        body_idxs = self.agent_body_idxs
        obs["body_idx"] = self._get_one_hot_body_idx()
        # Absolute position, orientation, linear and angular velocities
        torso_x_pos = sim.data.get_body_xpos("torso/0")[0]
        body_xpos = sim.data.body_xpos[body_idxs, :].copy()
        body_xpos[:, 0] -= torso_x_pos
        obs["body_xpos"] = body_xpos
        obs["body_xquat"] = sim.data.body_xquat[body_idxs, :].copy()
        obs["body_xvelp"] = sim.data.body_xvelp[body_idxs, :].copy()
        obs["body_xvelr"] = sim.data.body_xvelr[body_idxs, :].copy()

        # Relative position, orientation
        obs["body_pos"] = sim.model.body_pos[body_idxs, :].copy()
        obs["body_ipos"] = sim.model.body_ipos[body_idxs, :].copy()
        obs["body_iquat"] = sim.model.body_iquat[body_idxs, :].copy()
        obs["geom_quat"] = sim.model.geom_quat[self.agent_geom_idxs, :].copy()
        obs["geom_extremities"] = self.extremities(sim)

        # Hardware property
        obs["body_mass"] = sim.model.body_mass[body_idxs].copy()[:, np.newaxis]
        obs["body_shape"] = sim.model.geom_size[self.agent_geom_idxs, :2].copy()
        obs["body_friction"] = sim.model.geom_friction[self.agent_geom_idxs, 0:1].copy()
        return self._select_obs(obs)

    def get_joint_obs(self, sim):
        obs = {}
        qpos = sim.data.qpos.flat[7:].copy()
        qvel = sim.data.qvel.flat[6:].copy()

        joint_range = sim.model.jnt_range[1:, :].copy()
        qpos = (qpos - joint_range[:, 0]) / (joint_range[:, 1] - joint_range[:, 0])

        obs["qpos"] = qpos[:, np.newaxis]
        obs["qvel"] = qvel[:, np.newaxis]
        obs["jnt_pos"] = sim.model.jnt_pos[1:, :].copy()
        obs["joint_range"] = joint_range
        obs["joint_axis"] = sim.model.jnt_axis[1:, :].copy()
        obs["gear"] = sim.model.actuator_gear[:, 0:1].copy()
        obs["armature"] = sim.model.dof_armature[6:].copy()[:, np.newaxis]
        obs["damping"] = sim.model.dof_damping[6:].copy()[:, np.newaxis]

        return self._select_obs(obs)

    def _get_one_hot_body_idx(self):
        body_idxs = self.agent_body_idxs
        one_hot_encoding = np.zeros((len(body_idxs), cfg.MODEL.MAX_LIMBS))
        rows = list(range(0, len(body_idxs)))
        one_hot_encoding[rows, body_idxs] = 1
        return one_hot_encoding

    def _select_obs(self, obs):
        obs_to_ret = []
        for obs_type in cfg.MODEL.PROPRIOCEPTIVE_OBS_TYPES:
            if obs_type in obs:
                obs_to_ret.append(obs[obs_type])

        if len(obs_to_ret):
            return np.hstack(tuple(obs_to_ret))
        else:
            return None

    def combine_limb_joint_obs(self, limb_obs, joint_obs, env):
        # Create node centric observations where each node observation is
        # concatenation of limb features, joint features. The joint features
        # are concatenation of all hinge joints connecting the limb with it's
        # parent limb.
        num_limbs = len(self.agent_body_idxs)
        joint_obs_size = joint_obs.shape[1]
        # Two limbs can be connected by atmost 2 hinge joints
        joint_obs_padded = np.zeros((num_limbs, joint_obs_size * 2))
        joint_obs_padded = joint_obs_padded.reshape(-1, joint_obs_size)
        joint_obs_padded[self.joint_mask_for_node_graph, :] = joint_obs
        joint_obs_padded = joint_obs_padded.reshape(num_limbs, -1)
        if limb_obs is None:
            obs = joint_obs_padded
        else:
            obs = np.hstack((limb_obs, joint_obs_padded))

        if (cfg.MIRROR_DATA_AUG and env.metadata["mirrored"]):
            obs = obs[env.metadata["o_to_m"], :]
        return obs.flatten()

    def observation_step(self, env, sim):
        limb_obs = self.get_limb_obs(sim)
        joint_obs = self.get_joint_obs(sim)
        return {
            "proprioceptive": self.combine_limb_joint_obs(limb_obs, joint_obs, env),
            "edges": self.edges
        }

    def _add_fixed_cameras(self, worldbody):
        cameras = [
            cu.PATROL_VIEW,
            cu.TUNE_CAMERA,
        ]
        insert_pos = 1

        for spec in cameras:
            worldbody.insert(insert_pos, xu.camera_elem(spec))
            insert_pos += 1

    def _add_cameras(self, head):
        cameras = [
            cu.INCLINE_VIEW,
            cu.MANI_VIEW,
            cu.OBSTACLE_VIEW,
            cu.FT_VIEW,
            cu.VT_VIEW,
            cu.LEFT_VIEW,
            cu.TOP_DOWN,
            cu.FRONT_VIEW,
            cu.REAR_VIEW,
        ]
        insert_pos = 0
        for idx, child_elem in enumerate(head):
            if child_elem.tag == "camera":
                insert_pos = idx + 1
                break

        for spec in cameras:
            head.insert(insert_pos, xu.camera_elem(spec))
            insert_pos += 1

    ###########################################################################
    # Proprioceptive observations
    ###########################################################################
    def position(self, sim):
        pos = sim.data.qpos.flat.copy()
        pos = pos[self.agent_qpos_idxs]

        if not cfg.ENV.SKIP_SELF_POS:
            return pos
        # Ignores horizontal position to maintain translational invariance
        if cfg.HFIELD.DIM == 1:
            pos = pos[1:]
        else:
            # Skip the 7 DoFs of the free root joint
            pos = pos[7:]
        return pos

    def velocity(self, sim):
        vel = sim.data.qvel.flat.copy()
        return vel[self.agent_qvel_idxs]

    def imu_vel(self, sim):
        # Return torso acceleration, torso gyroscope and torso velocity
        return sim.data.sensordata[:9].copy()

    def touch(self, sim):
        # Return scalar force, each limb/torso has one touch sensor
        return sim.data.sensordata[12:].copy()

    def extremities(self, sim):
        """Returns limb positions in torso/0 frame."""
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        torso_pos = sim.data.get_body_xpos("torso/0")
        positions = [[0.0] * 3]  # Add torso position
        for site_name in self.limb_btm_sites:
            torso_to_limb = sim.data.get_site_xpos(site_name) - torso_pos
            positions.append(torso_to_limb.dot(torso_frame))
        extremities = np.vstack(positions)
        return extremities


def merge_agent_with_base(agent, ispath=True):
    base_xml = cfg.UNIMAL_TEMPLATE
    root_b, tree_b = xu.etree_from_xml(base_xml)
    root_a, tree_a = xu.etree_from_xml(agent, ispath=ispath)

    worldbody = root_b.findall("./worldbody")[0]
    agent_body = xu.find_elem(root_a, "body", "name", "torso/0")[0]

    # Update agent z pos based on starting terrain
    pos = xu.str2arr(agent_body.get("pos"))
    pos[2] += cfg.TERRAIN.SIZE[2]
    agent_body.set("pos", xu.arr2str(pos))
    worldbody.append(agent_body)

    actuator_a = root_a.findall("./actuator")[0]
    actuator_b = root_b.findall("./actuator")[0]
    agent_motors = xu.find_elem(actuator_a, "motor")
    actuator_b.extend(agent_motors)

    sensor_a = root_a.findall("./sensor")[0]
    sensor_b = root_b.findall("./sensor")[0]
    sensor_b.extend(list(sensor_a))
    return xu.etree_to_str(root_b)


def extract_agent_from_xml(xml_path):
    root, tree = xu.etree_from_xml(xml_path)
    agent = etree.Element("agent", {"model": "unimal"})
    unimal = xu.find_elem(root, "body", "name", "torso/0")[0]
    actuator = root.findall("./actuator")[0]
    sensor = root.findall("./sensor")[0]
    agent.append(unimal)
    agent.append(actuator)
    agent.append(sensor)
    agent = xu.etree_to_str(agent)
    return agent


def modify_xml_attributes(xml):
    root, tree = xu.etree_from_xml(xml, ispath=False)

    # Modify njmax and nconmax
    size = xu.find_elem(root, "size")[0]
    size.set("njmax", str(cfg.XML.NJMAX))
    size.set("nconmax", str(cfg.XML.NCONMAX))

    # Enable/disable filterparent
    flag = xu.find_elem(root, "flag")[0]
    flag.set("filterparent", str(cfg.XML.FILTER_PARENT))

    # Modify default geom params
    default_geom = xu.find_elem(root, "geom")[0]
    default_geom.set("condim", str(cfg.XML.GEOM_CONDIM))
    default_geom.set("friction", xu.arr2str(cfg.XML.GEOM_FRICTION))

    # Modify njmax and nconmax
    visual = xu.find_elem(root, "visual")[0]
    map_ = xu.find_elem(visual, "map")[0]
    map_.set("shadowclip", str(cfg.XML.SHADOWCLIP))

    return xu.etree_to_str(root)


def create_agent_xml(path):
    agent_xml = extract_agent_from_xml(path)
    xml = merge_agent_with_base(agent_xml, False)
    xml = modify_xml_attributes(xml)
    return xml
