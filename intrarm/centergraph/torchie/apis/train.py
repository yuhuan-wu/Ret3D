from functools import partial
import torch
from torch import nn

from centergraph.solver.fastai_optim import OptimWrapper


def flatten_model(m):
    return sum(map(flatten_model, m.children()), []) if len(list(m.children())) else [m]

def get_layer_groups(m):
    return [nn.Sequential(*flatten_model(m))]

def build_one_cycle_optimizer(model, optimizer_config):
    if optimizer_config.fixed_wd:
        optimizer_func = partial(
            torch.optim.Adam, betas=(0.9, 0.99), amsgrad=optimizer_config.amsgrad
        )
    else:
        optimizer_func = partial(torch.optim.Adam, amsgrad=optimizer_config.amsgrad)

    optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,   # TODO: CHECKING LR HERE !!!
        get_layer_groups(model),
        wd=optimizer_config.wd,
        true_wd=optimizer_config.fixed_wd,
        bn_wd=True,
    )

    return optimizer

