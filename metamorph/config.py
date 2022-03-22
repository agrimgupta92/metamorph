"""Configuration file (powered by YACS)."""

import copy
import os

from metamorph.yacs import CfgNode as CN

# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

# ----------------------------------------------------------------------------#
# XML template params
# ----------------------------------------------------------------------------#
# Refer mujoco docs for what each param does

_C.XML = CN()

_C.XML.NJMAX = 1000

_C.XML.NCONMAX = 200

_C.XML.GEOM_CONDIM = 3

_C.XML.GEOM_FRICTION = [0.7, 0.1, 0.1]

_C.XML.FILTER_PARENT = "enable"

_C.XML.SHADOWCLIP = 0.5

# ----------------------------------------------------------------------------#
# Unimal Env Options
# ----------------------------------------------------------------------------#
_C.ENV = CN()

_C.ENV.FORWARD_REWARD_WEIGHT = 1.0

_C.ENV.AVOID_REWARD_WEIGHT = 100.0

_C.ENV.CTRL_COST_WEIGHT = 0.0

_C.ENV.HEALTHY_REWARD = 0.0

_C.ENV.STAND_REWARD_WEIGHT = 0.0

_C.ENV.STATE_RANGE = (-100.0, 100.0)

_C.ENV.Z_RANGE = (-0.1, float("inf"))

_C.ENV.ANGLE_RANGE = (-0.2, 0.2)

_C.ENV.RESET_NOISE_SCALE = 5e-3

# Healthy reward is 1 if head_pos >= STAND_HEIGHT_RATIO * head_pos in
# the xml i.e the original height of the unimal.
_C.ENV.STAND_HEIGHT_RATIO = 0.5

# List of modules to add to the env. Modules will be added in the same order
_C.ENV.MODULES = ["Floor", "Agent"]

# Agent name if you are not using unimal but want to still use the unimal env
_C.ENV.WALKER_DIR = "./output/unimals_100/train"

# Agent name if you are not using unimal but want to still use the unimal env
_C.ENV.WALKERS = []

# Keys to keep in SelectKeysWrapper
_C.ENV.KEYS_TO_KEEP = []

# Skip position of free joint (or x root joint) in position observation for
# translation invariance
_C.ENV.SKIP_SELF_POS = False

# Specify task. Can be locomotion, manipulation
_C.ENV.TASK = "locomotion"

# Optional wrappers to add to task. Most wrappers for a task will eventually be
# hardcoded in make_env_task func. Put wrappers which you want to experiment
# with.
_C.ENV.WRAPPERS = []

# Task sampling strategy for multi-task envs. See multi_env_wrapper.py
_C.ENV.TASK_SAMPLING = "balanced_replay_buffer"

# For envs which change on each reset e.g. vt and mvt this should be true.
# For floor env this should be false, leads to 3x speed up.
_C.ENV.NEW_SIM_ON_RESET = True

# ----------------------------------------------------------------------------#
# Terrain Options
# ----------------------------------------------------------------------------#
# Attributes for x will be called length, y width and z height
_C.TERRAIN = CN()

# Size of the "floor/0" x, y, z
_C.TERRAIN.SIZE = [25, 20, 1]

_C.TERRAIN.START_FLAT = 2

_C.TERRAIN.CENTER_FLAT = 2

# Supported types of terrain obstacles
_C.TERRAIN.TYPES = ["gap", "jump"]

# Length of flat terrain
_C.TERRAIN.FLAT_LENGTH_RANGE = [9, 15, 2]

# Shared across avoid and jump
_C.TERRAIN.WALL_LENGTH = 0.1

# Length of terrain on which there will be hfield
_C.TERRAIN.HFIELD_LENGTH_RANGE = [4, 8, 4]

# Max height in case of slope profile
_C.TERRAIN.CURVE_HEIGHT_RANGE = [0.6, 1.2, 0.1]

_C.TERRAIN.BOUNDARY_WALLS = True

# Height of individual step
_C.TERRAIN.STEP_HEIGHT = 0.2

# Length of terrain on which there will be steps
_C.TERRAIN.STEP_LENGTH_RANGE = [12, 16, 4]

_C.TERRAIN.NUM_STEPS = 8

_C.TERRAIN.RUGGED_SQUARE_CLIP_RANGE = [0.2, 0.3, 0.1]

# Max height of bumps in bowl
_C.TERRAIN.BOWL_MAX_Z = 1.3

# Angle of incline for incline task
_C.TERRAIN.INCLINE_ANGLE = 0

# Vertical distance between the bottom most point of unimal and floor
_C.TERRAIN.FLOOR_OFFSET = 0.2
# ----------------------------------------------------------------------------#
# Objects Options
# ----------------------------------------------------------------------------#
# Attributes for x will be called length, y width and z height
_C.OBJECT = CN()

# Goal position, if empty each episode will have a different goal position. Or
# you can specify the position here. Only specify the x, y position.
_C.OBJECT.GOAL_POS = []

# Same as GOAL_POS
_C.OBJECT.BOX_POS = []

# Min distance from the walls to place the object
_C.OBJECT.PLACEMENT_BUFFER_LEN = 3

_C.OBJECT.PLACEMENT_BUFFER_WIDTH = 0

# Half len of square for close placement
_C.OBJECT.CLOSE_PLACEMENT_DIST = 10

# Min distance between agent and goal for success
_C.OBJECT.SUCCESS_MARGIN = 0.5

# Side len of the box
_C.OBJECT.BOX_SIDE = 0.5

# Number of obstacles for obstacle env
_C.OBJECT.NUM_OBSTACLES = 50

# Length of the obstacle box
_C.OBJECT.OBSTACLE_LEN_RANGE = [0.5, 1, 0.1]

# Width of the obstacle box
_C.OBJECT.OBSTACLE_WIDTH_RANGE = [0.5, 1, 0.1]

# Range of distance between successive object placements for forward_placement
_C.OBJECT.FORWARD_PLACEMENT_DIST = [10, 15]

# Typpe of object to manipulate can be box or ball
_C.OBJECT.TYPE = "box"

_C.OBJECT.BALL_RADIUS = 0.15

_C.OBJECT.BOX_MASS = 1.0

# ----------------------------------------------------------------------------#
# Hfield Options
# ----------------------------------------------------------------------------#
_C.HFIELD = CN()

# For planer walker type unimals 1 otherwise 2
_C.HFIELD.DIM = 2

# Slice of hfield given to agent as obs. [behind, front, right, left] or
# [-x, +x, -y, +y]
_C.HFIELD.OBS_SIZE = [1, 4, 4, 4]

_C.HFIELD.ADAPTIVE_OBS = False

# See _cal_hfield_bounds in hfield.py
_C.HFIELD.ADAPTIVE_OBS_SIZE = [0.10, 0.50, 1.5, 5.0]

# Pad hfiled for handling agents on edges of the terrain. Padding value should
# be greater than sqrt(2) * max(HFIELD.OBS_SIZE). As when you rotate the
# the hfield obs, square diagonal should fit inside padding.
_C.HFIELD.PADDING = 10

# Number representing that the terrain has gap in hfield obs
_C.HFIELD.GAP_DEPTH = -10

# Number of divisions in 1 unit for hfield, should be a multiple of 10
_C.HFIELD.NUM_DIVS = 10

# Viz the extreme points of hfield
_C.HFIELD.VIZ = False

# ----------------------------------------------------------------------------#
# Video Options
# ----------------------------------------------------------------------------#
_C.VIDEO = CN()

# Save video
_C.VIDEO.SAVE = False

# Frame width
_C.VIDEO.WIDTH = 640

# Frame height
_C.VIDEO.HEIGHT = 360

# FPS for saving
_C.VIDEO.FPS = 30

# --------------------------------------------------------------------------- #
# PPO Options
# --------------------------------------------------------------------------- #
_C.PPO = CN()

# Discount factor for rewards
_C.PPO.GAMMA = 0.99

# GAE lambda parameter
_C.PPO.GAE_LAMBDA = 0.95

# Hyperparameter which roughly says how far away the new policy is allowed to
# go from the old
_C.PPO.CLIP_EPS = 0.2

# Number of epochs (K in PPO paper) of sgd on rollouts in buffer
_C.PPO.EPOCHS = 8

# Batch size for sgd (M in PPO paper)
_C.PPO.BATCH_SIZE = 5120

# Value (critic) loss term coefficient
_C.PPO.VALUE_COEF = 0.5

# If KL divergence between old and new policy exceeds KL_TARGET_COEF * 0.01
# stop updates. Default value is high so that it's not used by default.
_C.PPO.KL_TARGET_COEF = 20.0

# Clip value function
_C.PPO.USE_CLIP_VALUE_FUNC = True

# Entropy term coefficient
_C.PPO.ENTROPY_COEF = 0.0

# Max timesteps per rollout
_C.PPO.TIMESTEPS = 2560

# Number of parallel envs for collecting rollouts
_C.PPO.NUM_ENVS = 32

# Learning rate ranges from BASE_LR to MIN_LR*BASE_LR according to the LR_POLICY
_C.PPO.BASE_LR = 3e-4
_C.PPO.MIN_LR = 0.0

# Learning rate policy select from {'cos', 'lin'}
_C.PPO.LR_POLICY = "cos"

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.PPO.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of iters
_C.PPO.WARMUP_ITERS = 5

# EPS for Adam/RMSProp
_C.PPO.EPS = 1e-5

# Value to clip the gradient via clip_grad_norm_
_C.PPO.MAX_GRAD_NORM = 0.5

# Total number of env.step() across all processes and all rollouts over the
# course of training
_C.PPO.MAX_STATE_ACTION_PAIRS = 1e8

# Iter here refers to 1 cycle of experience collection and policy update.
# Refer PPO paper. This is field is inferred see: calculate_max_iters()
_C.PPO.MAX_ITERS = -1

# Length of video to save while evaluating policy in num env steps. Env steps
# may not be equal to actual simulator steps. Actual simulator steps would be
# env_steps * frame_skip.
_C.PPO.VIDEO_LENGTH = 1000

# Path to load model from
_C.PPO.CHECKPOINT_PATH = ""

_C.PPO.EARLY_EXIT = False

_C.PPO.EARLY_EXIT_STATE_ACTION_PAIRS = 1e8

_C.PPO.EARLY_EXIT_MAX_ITERS = -1

# --------------------------------------------------------------------------- #
# Task sampling options
# --------------------------------------------------------------------------- #

_C.TASK_SAMPLING = CN()

_C.TASK_SAMPLING.EMA_ALPHA = 0.1

_C.TASK_SAMPLING.PROB_ALPHA = 1.0

_C.TASK_SAMPLING.AVG_TYPE = "ema"

# --------------------------------------------------------------------------- #
# Model Options
# --------------------------------------------------------------------------- #
_C.MODEL = CN()

# Type of actor critic model: ActorCritic
_C.MODEL.ACTOR_CRITIC = "ActorCritic"

_C.MODEL.LIMB_EMBED_SIZE = 128

_C.MODEL.JOINT_EMBED_SIZE = 128

# Max number of joints across all the envs
_C.MODEL.MAX_JOINTS = 7

# Max number of limbs across all the envs
_C.MODEL.MAX_LIMBS = 8

# Fixed std value
_C.MODEL.ACTION_STD = 0.9

# Use fixed or learnable std
_C.MODEL.ACTION_STD_FIXED = True

# Types of proprioceptive obs to include
_C.MODEL.PROPRIOCEPTIVE_OBS_TYPES = [
    "body_xpos", "body_xvelp", "body_xvelr", "body_xquat", "body_pos", "body_ipos", "body_iquat", "geom_quat", # limb
    "body_mass", "body_shape", # limb hardware
    "qpos", "qvel", "jnt_pos", # joint
    "joint_range", "joint_axis", "gear" # joint hardware
]

# Model specific observation types to keep
_C.MODEL.OBS_TYPES = ["proprioceptive", "edges", "obs_padding_mask", "act_padding_mask"]

# Observations to normalize via VecNormalize
_C.MODEL.OBS_TO_NORM = ["proprioceptive"]

# Wrappers to add specific to model
_C.MODEL.WRAPPERS = ["MultiUnimalNodeCentricObservation", "MultiUnimalNodeCentricAction"]

# --------------------------------------------------------------------------- #
# Transformer Options
# --------------------------------------------------------------------------- #
_C.MODEL.TRANSFORMER = CN()

# Number of attention heads in TransformerEncoderLayer (nhead)
_C.MODEL.TRANSFORMER.NHEAD = 2

# TransformerEncoderLayer (dim_feedforward)
_C.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 1024

# TransformerEncoderLayer (dropout)
_C.MODEL.TRANSFORMER.DROPOUT = 0.0

# Number of TransformerEncoderLayer in TransformerEncoder
_C.MODEL.TRANSFORMER.NLAYERS = 5

# Init for input embedding
_C.MODEL.TRANSFORMER.EMBED_INIT = 0.1

# Init for output decoder embodedding
_C.MODEL.TRANSFORMER.DECODER_INIT = 0.01

_C.MODEL.TRANSFORMER.DECODER_DIMS = []

_C.MODEL.TRANSFORMER.EXT_HIDDEN_DIMS = []

# Early vs late fusion of exterioceptive observation
_C.MODEL.TRANSFORMER.EXT_MIX = "none"

# Type of position embedding to use: None, learnt
_C.MODEL.TRANSFORMER.POS_EMBEDDING = "learnt"

# --------------------------------------------------------------------------- #
# Finetuning Options
# --------------------------------------------------------------------------- #
_C.MODEL.FINETUNE = CN()

# If true fine tune all the model params, if false fine tune only specific layer
_C.MODEL.FINETUNE.FULL_MODEL = False

# Name of layers to fine tune
_C.MODEL.FINETUNE.LAYER_SUBSTRING = []

# --------------------------------------------------------------------------- #
# Sampler (VecEnv) Options
# --------------------------------------------------------------------------- #
_C.VECENV = CN()

# Type of vecenv. DummyVecEnv is generally the fastest option for light weight
# envs. The fatest configuration is most likely DummyVecEnv coupled with DDP.
# Note: It is faster to have N dummyVecEnvs in DDP than having the same config
# via SubprocVecEnv.
_C.VECENV.TYPE = "SubprocVecEnv"

# Number of envs to run in series for SubprocVecEnv
_C.VECENV.IN_SERIES = 2

# --------------------------------------------------------------------------- #
# CUDNN options
# --------------------------------------------------------------------------- #
_C.CUDNN = CN()

_C.CUDNN.BENCHMARK = False
_C.CUDNN.DETERMINISTIC = True

# ----------------------------------------------------------------------------#
# Misc Options
# ----------------------------------------------------------------------------#
# Output directory
_C.OUT_DIR = "./output"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries. This is the only seed
# which will effect env variations.
_C.RNG_SEED = 1

# Name of the environment used for experience collection
_C.ENV_NAME = "Unimal-v0"

# Use GPU
_C.DEVICE = "cuda:0"

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters.
_C.LOG_PERIOD = 10

# Checkpoint period in iters. Refer LOG_PERIOD for meaning of iter
_C.CHECKPOINT_PERIOD = 100

# Evaluate the policy after every EVAL_PERIOD iters
_C.EVAL_PERIOD = -1

# Node ID for distributed runs
_C.NODE_ID = -1

# Number of nodes
_C.NUM_NODES = 1

# Unimal template path relative to the basedir
_C.UNIMAL_TEMPLATE = "./metamorph/envs/assets/unimal.xml"

# Save histogram weights
_C.SAVE_HIST_WEIGHTS = False

# Optional description for exp
_C.DESC = ""

# How to handle mjstep exception
_C.EXIT_ON_MJ_STEP_EXCEPTION = False

_C.MIRROR_DATA_AUG = False

def dump_cfg(cfg_name=None):
    """Dumps the config to the output directory."""
    if not cfg_name:
        cfg_name = _C.CFG_DEST
    cfg_file = os.path.join(_C.OUT_DIR, cfg_name)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def get_default_cfg():
    return copy.deepcopy(cfg)
