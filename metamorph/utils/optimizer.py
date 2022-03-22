"""Optimizer."""

import numpy as np
from metamorph.config import cfg


def lr_fun_cos(cur_iter):
    """Cosine schedule (cfg.PPO.LR_POLICY = 'cos')."""
    lr = 0.5 * (1.0 + np.cos(np.pi * cur_iter / cfg.PPO.MAX_ITERS))
    return (1.0 - cfg.PPO.MIN_LR) * lr + cfg.PPO.MIN_LR


def lr_fun_lin(cur_iter):
    """Linear schedule (cfg.PPO.LR_POLICY = 'lin')."""
    lr = 1.0 - cur_iter / cfg.PPO.MAX_ITERS
    return (1.0 - cfg.PPO.MIN_LR) * lr + cfg.PPO.MIN_LR


def get_lr_fun():
    """Retrieves the specified lr policy function"""
    lr_fun = "lr_fun_" + cfg.PPO.LR_POLICY
    assert lr_fun in globals(), "Unknown LR policy: " + cfg.PPO.LR_POLICY
    err_str = "exp lr policy requires PPO.MIN_LR to be greater than 0."
    assert cfg.PPO.LR_POLICY != "exp" or cfg.PPO.MIN_LR > 0, err_str
    return globals()[lr_fun]


def get_iter_lr(cur_iter):
    """Retrieves the lr for the given iter according to the policy."""
    # Get lr and scale by by BASE_LR
    lr = get_lr_fun()(cur_iter) * cfg.PPO.BASE_LR
    # Linear warmup
    if cur_iter < cfg.PPO.WARMUP_ITERS:
        alpha = cur_iter / cfg.PPO.WARMUP_ITERS
        warmup_factor = cfg.PPO.WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    return lr


def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def plot_lr_fun():
    """Visualizes lr function."""
    import matplotlib.pyplot as plt
    iters = list(range(cfg.PPO.MAX_ITERS))
    lrs = [get_iter_lr(iter_) for iter_ in iters]
    plt.plot(iters, lrs, ".-")
    plt.title("lr_policy: {}".format(cfg.PPO.LR_POLICY))
    plt.xlabel("iters")
    plt.ylabel("learning rate")
    plt.ylim(bottom=0)
    plt.savefig('./output/lr.png')
