# TODO!

import time
import sys
from typing import Dict
from argparse import Namespace
import copy
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module
from torch.autograd import Variable
from torch.utils.data import DataLoader


'''https://github.com/davda54/sam'''

import torch
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class FisherSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, eta=1, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FisherSAM, self).__init__(params, defaults)

        self.eta=eta
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, param_to_name, fisher_mats, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                param_name = param_to_name.get(p, "Unnamed parameter")
                # Compute the Fisher inverse and apply it to the gradient
                fisher_diag = fisher_mats[param_name].detach()  # Get Fisher diagonal for current parameter
                fisher_inv_grad = p.grad/(1 + self.eta*fisher_diag)   # F(θ)⁻¹ ∇l(θ)

                # Normalization term: √(∇l(θ) F(θ)⁻¹ ∇l(θ))
                grad_fisher_grad = torch.dot(p.grad.view(-1), fisher_inv_grad.view(-1))
                normalization_factor = torch.sqrt(grad_fisher_grad + 1e-12)

                # Apply normalization
                perturbation = fisher_inv_grad / (normalization_factor + 1e-12)

                # Scale perturbation by rho
                # print(torch.norm(perturbation - p.grad))
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * perturbation * group["rho"]
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # Get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # Do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # The closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_fisher_norm(self):
        """ Compute the norm of the gradient scaled by the inverse of the Fisher diagonal matrix """
        shared_device = self.param_groups[0]["params"][0].device  # Put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad / (self.fisher_mats[p] + 1e-12)).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def fim_diag(model: Module,
             data: Tensor,
             label: Tensor,
             device: torch.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")) -> Dict[str, Tensor]:
    """
    Computes the diagonal of the Fisher Information Matrix (FIM) using the gradient magnitude approximation over a given batch of data.
    
    Args:
        model (Module): The neural network model.
        data (Tensor): Input batch data.
        target (Tensor): Target labels for the batch.
        empirical (bool): Whether to compute the empirical Fisher (True) or expected Fisher (False).
        device (torch.device, optional): The device on which to perform computations (e.g., 'cpu' or 'cuda').
    
    Returns:
        Dict[str, Tensor]: A dictionary with parameter names as keys and their corresponding Fisher diagonal using gradient magnitude approximation.
    """
    precision_matrices = {}
    model_copy = copy.deepcopy(model).to(device)
    for n, p in model_copy.named_parameters():
        p.data.zero_()
        precision_matrices[n] = variable(p.data).to(device)

    model_copy.eval()
    model_copy.zero_grad()
    loss = torch.nn.functional.cross_entropy(model_copy(data), label)
    loss.backward()

    for n, p in model_copy.named_parameters():
        precision_matrices[n].data += p.grad.data ** 2

    precision_matrices = {n: p for n, p in precision_matrices.items()}
    return precision_matrices


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)