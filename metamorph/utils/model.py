import torch
import torch.nn as nn


def w_init(module, gain=1):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


def make_mlp(dim_list):
    init_ = lambda m: w_init(m)

    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(init_(nn.Linear(dim_in, dim_out)))
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def make_mlp_default(dim_list, final_nonlinearity=True, nonlinearity="relu"):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if nonlinearity == "relu":
            layers.append(nn.ReLU())
        elif nonlinearity == "tanh":
            layers.append(nn.Tanh())

    if not final_nonlinearity:
        layers.pop()
    return nn.Sequential(*layers)


def num_params(model, only_trainable=True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = model.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)
