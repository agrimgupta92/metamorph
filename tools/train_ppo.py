import argparse
import os
import sys

import torch

from metamorph.algos.ppo.ppo import PPO
from metamorph.config import cfg
from metamorph.config import dump_cfg
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu


def set_cfg_options():
    calculate_max_iters()
    maybe_infer_walkers()
    calculate_max_limbs_joints()


def calculate_max_limbs_joints():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    num_joints, num_limbs = [], []

    metadata_paths = []
    for agent in cfg.ENV.WALKERS:
        metadata_paths.append(os.path.join(
            cfg.ENV.WALKER_DIR, "metadata", "{}.json".format(agent)
        ))

    for metadata_path in metadata_paths:
        metadata = fu.load_json(metadata_path)
        num_joints.append(metadata["dof"])
        num_limbs.append(metadata["num_limbs"] + 1)

    # Add extra 1 for max_joints; needed for adding edge padding
    cfg.MODEL.MAX_JOINTS = max(num_joints) + 1
    cfg.MODEL.MAX_LIMBS = max(num_limbs) + 1


def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )


def maybe_infer_walkers():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    # Only infer the walkers if this option was not specified
    if len(cfg.ENV.WALKERS):
        return

    cfg.ENV.WALKERS = [
        xml_file.split(".")[0]
        for xml_file in os.listdir(os.path.join(cfg.ENV.WALKER_DIR, "xml"))
    ]


def get_hparams():
    hparam_path = os.path.join(cfg.OUT_DIR, "hparam.json")
    # For local sweep return
    if not os.path.exists(hparam_path):
        return {}

    hparams = {}
    varying_args = fu.load_json(hparam_path)
    flatten_cfg = swu.flatten(cfg)

    for k in varying_args:
        hparams[k] = flatten_cfg[k]

    return hparams


def cleanup_tensorboard():
    tb_dir = os.path.join(cfg.OUT_DIR, "tensorboard")

    # Assume there is only one sub_dir and break when it's found
    for content in os.listdir(tb_dir):
        content = os.path.join(tb_dir, content)
        if os.path.isdir(content):
            break

    # Return if no dir found
    if not os.path.isdir(content):
        return

    # Move all the event files from sub_dir to tb_idr
    for event_file in os.listdir(content):
        src = os.path.join(content, event_file)
        dst = os.path.join(tb_dir, event_file)
        fu.move_file(src, dst)

    # Delete the sub_dir
    os.rmdir(content)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="See morphology/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ppo_train():
    su.set_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    torch.set_num_threads(1)
    PPOTrainer = PPO()
    PPOTrainer.train()
    hparams = get_hparams()
    PPOTrainer.save_rewards(hparams=hparams)
    PPOTrainer.save_model()
    cleanup_tensorboard()


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    # Set cfg options which are inferred
    set_cfg_options()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Save the config
    dump_cfg()
    ppo_train()


if __name__ == "__main__":
    main()
