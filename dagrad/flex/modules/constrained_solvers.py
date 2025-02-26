import copy
from typing import List, Optional, Union
import numpy as np
import torch
# import tqdm as tqdm
from tqdm.auto import tqdm

from dagrad.flex.modules.models import MLP

class ConstrainedSolver:
    def __init__(self):
        self.device = None
        self.dtype = None
        self.verbose = None
        self.suppress_warnings = None

    def __call__(self, *args, **kwds):
        return self.solve(*args, **kwds)

    def __repr__(self):
        return f"Constrained Solver: {self.__class__.__name__}"

    def solve(self):
        raise NotImplementedError


class PathFollowing(ConstrainedSolver):
    def __init__(
        self,
        num_iter: int,
        mu_init: float,
        mu_scale: float,
        logdet_coeff: Optional[float] = None,
        num_steps: Optional[Union[List[int], int]] = None,
        weight_decay: Union[List[float], float] = 0,
        l1_coeff: Union[List[float], float] = 0,
    ):
        super().__init__()

        self.num_iter = num_iter
        self.num_steps = num_steps
        self.mu = mu_init
        self.mu_scale = mu_scale
        self.s = logdet_coeff
        self.weight_decay = weight_decay
        self.l1_coeff = l1_coeff

    def validate_inputs(self):
        if isinstance(self.num_steps, list):
            if len(self.num_steps) < self.num_iter:
                missing_steps = self.num_iter - len(self.num_steps)
                self.vwarn(
                    f"Using the first value from num_steps for the first {missing_steps} iterations"
                )
                self.num_steps = missing_steps * [self.num_steps[0]] + self.num_steps
        elif isinstance(self.num_steps, (int, float)):
            self.num_steps = self.num_iter * [self.num_steps]
        else:
            self.vwarn(
                "num_steps is None, will use the default value from Unconstrained Solver"
            )
            if self.unconstrained_solver.num_steps is None:
                raise ValueError("num_steps is None in Unconstrained Solver")
            else:
                self.num_steps = self.num_iter * [self.unconstrained_solver.num_steps]

        if isinstance(self.s, list):
            missing_s = self.num_iter - len(self.s)
            if len(self.s) < self.num_iter:
                self.vwarn(
                    f"Using the last value from s for the last {missing_s} iterations"
                )
                self.s = self.s + missing_s * [self.s[-1]]
        elif isinstance(self.s, (int, float)):
            self.s = self.num_iter * [self.s]
        else:
            ValueError("s should be a list, int, or float.")
        
        if self.l1_coeff > 0:
            # make sure the function l1_loss is implemented in the model
            if not hasattr(self.model, "l1_loss"):
                raise ValueError("Model does not have l1_loss method")

    def solve(self, dataset, model: MLP, unconstrained_solver, loss_fn, dag_fn):
        torch.set_default_dtype(self.dtype)
        self.model = model
        self.unconstrained_solver = unconstrained_solver
        self.vprint("Using Path Following Scheme to Solve the DAG constrained Problem")
        self.validate_inputs()

        for i in tqdm(range(self.num_iter)):
            dag_fn.s = self.s[i]  # used only for logdet function

            def new_loss(target):
                # total_loss = loss_fn(output, target)
                weights, biases = model.get_parameters()
                total_loss = - torch.mean(model.compute_log_likelihood(target, weights, biases))
                if self.l1_coeff > 0:
                    total_loss += self.l1_coeff * model.l1_loss()
                if self.weight_decay > 0:
                    l2_loss = torch.tensor(0.0).to(self.device)
                    for param in model.parameters():
                        l2_loss += torch.sum(param**2)
                    total_loss += 0.5 * self.weight_decay * l2_loss
                return self.mu * total_loss + dag_fn(model)

            success = False
            model_copy = copy.deepcopy(model)
            lr_scale, lr_scheduler = None, False

            while success is False:
                self.vprint(
                    f"\nSolving unconstrained problem #{i+1} with mu={self.mu}, s={dag_fn.s} for {self.num_steps[i]} steps"
                )
                unconstrained_solver.num_steps = self.num_steps[i]
                success = unconstrained_solver(
                    dataset, model, new_loss, dag_fn, lr_scale, lr_scheduler
                )

                if success is False:
                    model.load_state_dict(
                        model_copy.state_dict().copy()
                    )  # .to(device)?
                    lr_scale = 0.5 if lr_scale is None else 0.5 * lr_scale
                    lr_scheduler = True
                    if lr_scale < 1e-6:
                        break  # lr is too small
                    dag_fn.s = 1  # only useful when dag_fn is the logdet function
            self.mu *= self.mu_scale

def compute_loss(x, mask, model: MLP, weights, biases, extra_params,mean_std=False):
    # TODO: add param
    """
    Compute the loss. If intervention is perfect and known, remove
    the intervened targets from the loss with a mask.
    """
    # if intervention and intervention_type == "perfect" and intervention_knowledge =="known":
    #     log_likelihood = model.compute_log_likelihood(x, weights, biases, extra_params)
    #     log_likelihood = torch.sum(log_likelihood * mask, dim=0) / mask.size(0)
    # else:
    log_likelihood = model.compute_log_likelihood(x, weights, biases,
                                                  extra_params, mask=mask)
    log_likelihood = torch.sum(log_likelihood, dim=0) / mask.size(0)
    loss = - torch.mean(log_likelihood)

    if not mean_std:
        return loss
    else:
        joint_log_likelihood = torch.mean(log_likelihood * mask, dim=1)
        return loss, torch.sqrt(torch.var(joint_log_likelihood) / joint_log_likelihood.size(0))
    
def sample(target, batch_size):
    """
    Sample without replacement `batch_size` examples from the data and
    return the corresponding masks and regimes
    :param int batch_size: number of samples to sample
    :return: samples, masks, regimes
    """
    random = np.random.RandomState()
    num_samples = target.shape[0]
    sample_idxs = random.choice(np.arange(int(num_samples)), size=(int(batch_size),), replace=False)
    samples = target[torch.as_tensor(sample_idxs).long()]
    # if self.intervention:
    #     masks = self.convert_masks(sample_idxs)
    #     regimes = self.regimes[torch.as_tensor(sample_idxs).long()]
    # else:
    masks = torch.ones_like(samples)
    regimes = None
    return samples, masks, regimes


class AugmentedLagrangian(ConstrainedSolver):
    def __init__(
        self,
        num_iter: int = 20,
        num_steps: Optional[Union[List[int], int]] = None,
        alpha_init: float = 0.0,
        rho_init: float = 1.0,
        rho_scale: float = 10.0,
        rho_max: float = 1e8,
        weight_decay: float = 0.0,
        l1_coeff: float = 0.0,
        h_tol: float = 1e-8,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.num_steps = num_steps
        self.alpha_multiplier = alpha_init
        self.rho = rho_init
        self.rho_scale = rho_scale
        self.rho_max = rho_max
        self.weight_decay = weight_decay
        self.l1_coeff = l1_coeff
        self.h_tol = h_tol

    def validate_inputs(self):
        if isinstance(self.num_steps, list):
            if len(self.num_steps) < self.num_iter:
                missing_steps = self.num_iter - len(self.num_steps)
                self.vwarn(f"Using the first value from num_steps for the first {missing_steps} iterations")
                self.num_steps = missing_steps * [self.num_steps[0]] + self.num_steps
        elif isinstance(self.num_steps, (int, float)):
            self.num_steps = self.num_iter * [self.num_steps]
        else:
            self.vwarn("num_steps is None, will use default value from Unconstrained Solver")
            if self.unconstrained_solver.num_steps is None:
                raise ValueError("num_steps is None in Unconstrained Solver")
            else:
                self.num_steps = self.num_iter * [self.unconstrained_solver.num_steps]
        
        if self.l1_coeff > 0:
            # make sure the function l1_loss is implemented in the model
            if not hasattr(self.model, "l1_loss"):
                raise ValueError("Model does not have l1_loss method")

    def solve(self, dataset, model: MLP, unconstrained_solver, loss_fn, dag_fn, omega_lambda=1e-4, stop_crit_win=100):
        torch.set_default_dtype(self.dtype)
        self.model = model
        self.unconstrained_solver = unconstrained_solver
        self.vprint("Using Augmented Lagrangian Method to Solve the DAG constrained Problem")
        self.validate_inputs()

        end = False

        not_nlls = []  # Augmented Lagrangrian minus (pseudo) NLL
        aug_lagrangians_val = []
        nlls_val = []  # NLL on validation
        hs = []

        with torch.no_grad():
            full_adjacency = torch.ones((model.d, model.d)) - torch.eye(model.d)
            constraint_normalization = dag_fn(full_adjacency).item()


        for i in range(self.num_iter):
            if end:
                continue
            # while self.rho < self.rho_max:
            def augmented_loss(target):
                model.train()
                # Original loss
                # loss = loss_fn(output, target)
                x, mask, regime = sample(target,64)
                weights, biases = model.get_parameters()
                loss = compute_loss(x, mask, model, weights, biases, model.extra_params)
                # print(f'total loss: {total_loss}')
                model.eval()

                # DAG constraint
                w_adj = model.adj()
                h = dag_fn(w_adj) / constraint_normalization
                hs.append(h.item())
                
                reg = self.l1_coeff * model.compute_penalty([w_adj], p=1)
                reg /= w_adj.shape[0]**2
                reg_interv = torch.tensor(0)

                lagrangian = loss + reg + reg_interv + self.alpha_multiplier * h
                augmentation = h ** 2

                aug_lagrangian = lagrangian + 0.5 * self.rho * augmentation

                # Augmented Lagrangian terms
                # alm_term = self.alpha_multiplier * h + 0.5 * self.rho * h**2
                # print(f'loss is {loss} and alm term is {alm_term}')
                # alm_term = 0.5 * self.mu * h ** 2 + self.lambda_param * h
                not_nlls.append(reg.item() + 0.5 * self.rho * h.item() ** 2 + self.alpha_multiplier * h.item())
                # print(f'at iteration {i} loss is {loss} and alm term is {alm_term} and h is {h} and rho is {self.rho}')
                return aug_lagrangian

            success = False
            model_copy = model.state_dict().copy()
            lr_scale = None
            lr_scheduler = False

            while not success:
                self.vprint(
                    f"\nSolving ALM iteration #{i+1} with lambda={self.alpha_multiplier:.3f}, "
                    f"rho={self.rho:.3f} for {self.num_steps[i]} steps"
                )
                
                unconstrained_solver.num_steps = self.num_steps[i]
                success = unconstrained_solver(
                    dataset, model, augmented_loss, dag_fn, lr_scale, lr_scheduler
                )

                if not success:
                    model.load_state_dict(model_copy)
                    lr_scale = 0.5 if lr_scale is None else 0.5 * lr_scale
                    lr_scheduler = True
                    if lr_scale < 1e-6:
                        break
            
            if i % stop_crit_win == 0:
                with torch.no_grad():
                    loss_val = augmented_loss(dataset)
                    nlls_val.append(loss_val)
                    aug_lagrangians_val.append([i, loss_val + not_nlls[-1]])
                    
            if i >= 2 * stop_crit_win and i % (2 * stop_crit_win) == 0:
                t0, t_half, t1 = aug_lagrangians_val[-3][1], aug_lagrangians_val[-2][1], aug_lagrangians_val[-1][1]

                # if the validation loss went up and down, do not update lagrangian and penalty coefficients.
                if not (min(t0, t1) < t_half < max(t0, t1)):
                    delta_lambda = -np.inf
                else:
                    delta_lambda = (t1 - t0) / stop_crit_win
            else:
                delta_lambda = -np.inf  # do not update lambda nor mu
            
            if hs[-1] > self.h_tol:
                if abs(delta_lambda) < omega_lambda or delta_lambda > 0:
                    self.alpha_multiplier += self.rho * hs[-1]
                    # print("Updated lambda to {}".format(lamb))

                    # Did the constraint improve sufficiently?
                    # if len(hs) >= 2:
                    if len(hs) >= 2:
                        if hs[-1] > 0.9 * hs[-2]:
                            self.rho *= self.rho_scale
                            # print("Updated mu to {}".format(mu))

                    # little hack to make sure the moving average is going down.
                    with torch.no_grad():
                        gap_in_not_nll = 0.5 * self.rho * hs[-1] ** 2 + self.alpha_multiplier * hs[-1] - not_nlls[-1]
                        # aug_lagrangian_ma[iter + 1] += gap_in_not_nll
                        aug_lagrangians_val[-1][1] += gap_in_not_nll


                # if opt.optimizer == "rmsprop":
                # optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.lr_reinit)
                # else:
                #     optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr_reinit)

            else:
                with torch.no_grad():
                    w_adj = model.adj()
                    to_keep = (w_adj > 0).type(torch.Tensor)
                    model.adjacency *= to_keep
                end = True

                # if h_new > 0.9 * h:
                #     self.rho *= self.rho_scale
                # else:
                #     break
            # h = h_new
            # self.vprint(f"Current h(W): {h:.4f}")
            # self.alpha_multiplier += self.rho * h
            # if h <= self.h_tol or self.rho >= self.rho_max:
            #     end = True



# TODO: Implement TOPOSolver (Bilevel)