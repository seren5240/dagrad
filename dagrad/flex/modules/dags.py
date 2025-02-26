import numpy as np
import torch
from dagrad.flex.modules.models import MLP
from dagrad.utils.utils import simulate_dag
from warnings import warn
from . import functional as F
from scipy.linalg import expm

__all__ = ["DagFn", "Exp", "LogDet", "Poly", "ExpAbs", "LogDetAbs", "PolyAbs"]


class DagFn:
    def __init__(self) -> None:
        pass

    def __call__(self, W):
        return self.eval(W)

    def __repr__(self):
        return f"DAG Function: {self.__class__.__name__}"

    def eval(self): ...

    def validate_acyclicity(self):
        for d in [5, 10, 20, 50]:
            for graph_type in ["ER", "SF"]:
                B = simulate_dag(d, s0=int(d / 2), graph_type=graph_type)
                W = torch.tensor(B * np.random.randn(d, d))
                h = self.eval(W)
                if h > 1e-6:
                    warn(
                        f"Acyclicity Function might be incorrect. h = {h}, graph_type = {graph_type}, d = {d}"
                    )


class Exp(DagFn):
    def eval(self, W):
        return F.exp(W)


class LogDet(DagFn):
    def __init__(self, s: float = 1.0) -> None:
        super().__init__()
        self.s = s

    def eval(self, W):
        return F.log_det(W, s=self.s)


class Poly(DagFn):
    def eval(self, W):
        return F.poly(W)


class ExpAbs(DagFn):
    def eval(self, W):
        return F.exp_abs(W)


class LogDetAbs(DagFn):
    def __init__(self, s: float = 1.0) -> None:
        super().__init__()
        self.s = s

    def eval(self, W):
        return F.log_det_abs(W, s=self.s)


class PolyAbs(DagFn):
    def eval(self, W):
        return F.poly_abs(W)

class TrExpScipy(torch.autograd.Function):
    """
    autograd.Function to compute trace of an exponential of a matrix
    """

    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            # send tensor to cpu in numpy format and compute expm using scipy
            expm_input = expm(input.detach().cpu().numpy())
            # transform back into a tensor
            expm_input = torch.as_tensor(expm_input)
            if input.is_cuda:
                # expm_input = expm_input.cuda()
                assert expm_input.is_cuda
            # save expm_input to use in backward
            ctx.save_for_backward(expm_input)

            # return the trace
            return torch.trace(expm_input)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            (expm_input,) = ctx.saved_tensors
            return expm_input.t() * grad_output

class DCDI_h(DagFn):
    def eval(self, w_adj):
            assert (w_adj >= 0).detach().cpu().numpy().all()
            h = TrExpScipy.apply(w_adj) - w_adj.shape[0]
            return h