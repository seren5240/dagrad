import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from dagrad.flex.modules.torchkit import SigmoidFlow, log_normal

from ...utils.topo_utils import create_new_topo, create_Z


class LocallyConnected(nn.Module):
    """
    Implements a local linear layer
    """

    def __init__(
        self,
        num_linear: int,
        input_features: int,
        output_features: int,
        bias: bool = True,
    ):
        r"""
        Parameters
        ----------
        num_linear : int
            num of local linear layers, i.e.
        input_features : int
            m1
        output_features : int
            m2
        bias : bool, optional
            Whether to include bias or not. Default: ``True``.


        Attributes
        ----------
        weight : [d, m1, m2]
        bias : [d, m2]
        """
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.zeros(num_linear, output_features, input_features)
        )
        # print(f'self.weight initialized to {self.weight}')
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_linear, output_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.input_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Implements the forward pass of the layer.

        Parameters
        ----------
        input : torch.Tensor
            Shape :math:`(n, d, m1)`

        Returns
        -------
        torch.Tensor
            Shape :math:`(n, d, m2)`
        """
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = input.unsqueeze(dim=2) @ self.weight.unsqueeze(dim=0)
        out = out.squeeze(dim=2)
        print(f'out dimension is {out.shape}, bias dimension is {self.bias.shape}')
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self) -> str:
        """
        Returns a string with extra information from the layer.
        """
        return f"num_linear={self.num_linear}, in_features={self.input_features}, out_features={self.output_features}, bias={self.bias is not None}"


class LinearModel(nn.Module):
    def __init__(self, d, bias=False, dtype=torch.double):
        super().__init__()
        self.W = nn.Linear(d, d, bias=bias, dtype=dtype)
        nn.init.zeros_(self.W.weight)
        if bias:
            nn.init.zeros_(self.W.bias)

    def forward(self, x):
        return self.W(x)

    def adj(self):
        return self.W.weight.T
    
    def l1_loss(self):
        return self.W.weight.abs().sum()


class LogisticModel(nn.Module):
    def __init__(self, d, bias=False, dtype=torch.double):
        super(LogisticModel, self).__init__()
        self.W = nn.Linear(d, d, bias=bias, dtype=dtype)
        nn.init.zeros_(self.W.weight)
        if bias:
            nn.init.zeros_(self.W.bias)

    def forward(self, x):
        return torch.sigmoid(self.W(x))

    def adj(self):
        return self.W.weight.T
    
    def l1_loss(self):
        return self.W.weight.abs().sum()


class MLP(nn.Module):
    def __init__(
        self, dims, num_layers, hid_dim, activation="sigmoid", bias=True, dtype=torch.float64
    ) -> None:
        torch.set_default_dtype(dtype)
        super().__init__()
        # assert (
        #     len(dims) >= 2 and dims[-1] == 1
        # ), "Invalid dimension size or output dimension."
        self.d = dims[0]
        self.dims = dims
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.bias = bias
        # self.layers = nn.ModuleList()

        self.adjacency = torch.ones((self.d, self.d)) - torch.eye(
            self.d
        )
        self.gumbel_adjacency = GumbelAdjacency(self.d)

        if activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError("Activation function not supported.")

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        # self.fc2 = nn.ModuleList()
        for k in range(self.num_layers + 1):
            in_dim = self.hid_dim
            out_dim = self.hid_dim
            if k == 0:
                in_dim = self.d
            if k == self.num_layers:
                out_dim = dims[1]
            self.weights.append(nn.Parameter(torch.zeros(self.d, out_dim, in_dim)))
            self.biases.append(nn.Parameter(torch.zeros(self.d, out_dim)))

        self.reset_params()

        extra_params = np.ones((self.d,))
        np.random.shuffle(extra_params)
        # each element in the list represents a variable, the size of the element is the number of extra_params per var
        self.extra_params = nn.ParameterList()
        for extra_param in extra_params:
            self.extra_params.append(nn.Parameter(torch.tensor(np.log(extra_param).reshape(1)).type(torch.Tensor)))


    @staticmethod
    def make_hook_function(d):
        def hook_function(grad):
            grad_clone = grad.clone()
            grad_clone = grad_clone.view(d, -1, d)
            for i in range(d):
                grad_clone[i, :, i] = 0.0
            grad_clone = grad_clone.view(-1, d)
            return grad_clone

        return hook_function

    def compute_penalty(self, list_, p=2, target=0.):
        penalty = 0
        for m in list_:
            penalty += torch.norm(m - target, p=p) ** p
        return penalty
    
    def forward_given_params(self, x, weights, biases):
        """

        :param x: batch_size x num_vars
        :param weights: list of lists. ith list contains weights for ith MLP
        :param biases: list of lists. ith list contains biases for ith MLP
        :return: batch_size x num_vars * num_params, the parameters of each variable conditional
        """
        bs = x.size(0)
        # num_zero_weights = 0
        # print(f'num_layers is {num_layers}, len weights are {len(weights)} and len biases is {len(biases)}')
        for layer in range(self.num_layers + 1):
            # apply affine operator
            if layer == 0:
                M = self.gumbel_adjacency(bs)
                adj = self.adjacency.unsqueeze(0)
                x = torch.einsum("tij,bjt,ljt,bj->bti", weights[layer], M, adj, x) 
                x = x + biases[layer]
            else:
                x = torch.einsum("tij,btj->bti", weights[layer], x) + biases[layer]

            # count num of zeros
            # num_zero_weights += weights[k].numel() - weights[k].nonzero().size(0)

            # apply non-linearity
            if layer != self.num_layers:
                x = F.leaky_relu(x) # if self.nonlin == "leaky-relu" else torch.sigmoid(x)

        return torch.unbind(x, 1)

    def adj(self):
        """Get weighted adjacency matrix"""
        return self.gumbel_adjacency.get_proba() * self.adjacency

    def reset_params(self):
        with torch.no_grad():
            for node in range(self.d):
                for i, w in enumerate(self.weights):
                    w = w[node]
                    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('leaky_relu'))
                for i, b in enumerate(self.biases):
                    b = b[node]
                    b.zero_()

    def get_parameters(self):
        params = []
        
        weights = []
        for w in self.weights:
            weights.append(w)
        params.append(weights)

        biases = []
        for b in self.biases:
            biases.append(b)
        params.append(biases)

        return tuple(params)

    def get_distribution(self, dp):
        return torch.distributions.normal.Normal(dp[0], dp[1])

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev

    
    def compute_log_likelihood(self, x, weights, biases, detach=False):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :param weights: list of tensor that are coherent with self.weights
        :param biases: list of tensor that are coherent with self.biases
        :return: (batch_size, num_vars) log-likelihoods
        """
        density_params = self.forward_given_params(x, weights, biases)
        # print(f'density params are {density_params}')

        # if len(extra_params) != 0:
        extra_params = self.transform_extra_params(self.extra_params)
        log_probs = []
        for i in range(self.d):
            density_param = list(torch.unbind(density_params[i], 1))
            if len(extra_params) != 0:
                density_param.extend(list(torch.unbind(extra_params[i], 0)))
            conditional = self.get_distribution(density_param)
            x_d = x[:, i].detach() if detach else x[:, i]
            log_probs.append(conditional.log_prob(x_d).unsqueeze(1))

        return torch.cat(log_probs, 1)


class TopoMLP(nn.Module):
    def __init__(self, dims, activation="sigmoid", bias=True, dtype=torch.float64):
        super(TopoMLP, self).__init__()
        assert (
            len(dims) >= 2 and dims[-1] == 1
        ), "Invalid dimension size or output dimension."
        self.d = dims[0]
        self.d1 = dims[1]
        self.dtype = dtype
        self.dims = dims
        self.depth = len(dims)
        self.layerlist = nn.ModuleList()
        self.bias = bias
        self._create_layerlist()
        if activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError("Invalid activation function.")
        if self.bias is True:
            self.biasList = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1, self.d1).to(dtype=self.dtype))
                    for _ in range(self.d)
                ]
            )

    def _create_layerlist(self):
        # Initialize the first layer separately due to its unique structure
        layer0 = nn.ModuleList(
            [
                nn.Linear(1, self.d1, bias=False).to(dtype=self.dtype)
                for _ in range(self.d * self.d)
            ]
        )
        self.layerlist.append(layer0)
        # For subsequent layers, use a more streamlined approach
        for i in range(1, self.depth - 1):
            layers = nn.ModuleList(
                [
                    nn.Linear(self.dims[i], self.dims[i + 1], bias=self.bias).to(
                        dtype=self.dtype
                    )
                    for _ in range(self.d)
                ]
            )
            self.layerlist.append(layers)

    def set_all_para_grad_True(self):
        for param in self.parameters():
            param.requires_grad = True

    # set zero entry and set gradient non-updated for layer0!!!
    def reset_by_topo(self, topo):
        self.set_all_para_grad_True()
        Z = create_Z(topo)
        edge_abs_idx = np.argwhere(Z)
        with torch.no_grad():
            for idx in edge_abs_idx:
                linear_idx = int(idx[0] + self.d * idx[1])
                self.layerlist[0][linear_idx].weight.fill_(0)
                self.layerlist[0][linear_idx].weight.requires_grad = False

    def _forward_i(self, x, ith):
        # Improved forward pass to reduce complexity

        layer0_weights = torch.cat(
            [
                self.layerlist[0][ll].weight
                for ll in range(self.d * ith, self.d * (ith + 1))
            ],
            dim=1,
        ).T
        if self.bias:
            x = torch.mm(x, layer0_weights) + self.biasList[ith]
        else:
            x = torch.mm(x, layer0_weights)
        for ii in range(1, self.depth - 1):
            x = F.sigmoid(x)  # Consider using F.relu(x) for ReLU activation
            x = self.layerlist[ii][ith](x)
        return x

    #

    def forward(self, x):  # [n,d] ->[n,d]
        x = x.to(dtype=self.dtype)
        output = [self._forward_i(x, ii) for ii in range(self.d)]
        return torch.cat(output, dim=1)

    @torch.no_grad()
    def freeze_grad_f_i(self, ith):
        # freeze all the gradient of all the parameters related to f_i
        for k in range(self.d):
            self.layerlist[0][int(k + self.d * ith)].weight.requires_grad = False
        if self.bias:
            self.biasList[ith].requires_grad = False
        for i in range(1, self.depth - 1):
            self.layerlist[i][ith].weight.requires_grad = False
            if self.bias:
                self.layerlist[i][ith].bias.requires_grad = False

    def update_nn_by_topo(self, topo, index):
        # update the zero constraint and freeze corresponding gradient update
        i, j = index
        wherei, wherej = topo.index(i), topo.index(j)
        topo0 = create_new_topo(topo.copy(), index, opt=1)

        self.reset_by_topo(topo=topo0)
        freeze_idx = [
            oo for oo in range(self.d) if oo not in topo0[wherej : (wherei + 1)]
        ]
        if freeze_idx:
            for ith in freeze_idx:
                self.freeze_grad_f_i(ith)

    def layer0_l1_reg(self):
        return sum(torch.sum(torch.abs(vec.weight)) for vec in self.layerlist[0])

    def l2_reg(self):
        if self.bias:
            return sum(
                torch.sum(vec.weight**2) for layer in self.layerlist for vec in layer
            ) + sum(torch.sum(vec**2) for vec in self.biasList)
        else:
            return sum(
                torch.sum(vec.weight**2) for layer in self.layerlist for vec in layer
            )

    @torch.no_grad()
    def get_gradient_F(self):
        G_grad = torch.zeros(self.d**2, dtype=self.dtype)
        for count, vec in enumerate(self.layer0):
            G_grad[count] = torch.norm(vec.weight.grad, p=2)
        G_grad = torch.reshape(G_grad, (self.d, self.d)).t()
        return G_grad.numpy()

    @torch.no_grad()
    def layer0_to_adj(self):
        W = torch.zeros((self.d * self.d), dtype=self.dtype)
        for count, vec in enumerate(self.layerlist[0]):
            # W[count] = torch.sqrt(torch.sum(vec.weight ** 2))
            W[count] = torch.norm(vec.weight.data, p=2)
        W = torch.reshape(W, (self.d, self.d)).t()

        return W.numpy()

def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)
    
def gumbel_sigmoid(log_alpha, uniform, bs, tau=1, hard=False):
    shape = tuple([bs] + list(log_alpha.size()))
    logistic_noise = sample_logistic(shape, uniform)

    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).type(torch.Tensor)

        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft

    else:
        y = y_soft

    return y


class GumbelAdjacency(torch.nn.Module):
    """
    Random matrix M used for the mask. Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    :param int num_vars: number of variables
    """
    def __init__(self, num_vars):
        super(GumbelAdjacency, self).__init__()
        self.num_vars = num_vars
        self.log_alpha = torch.nn.Parameter(torch.zeros((num_vars, num_vars)))
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.reset_parameters()

    def forward(self, bs, tau=1, drawhard=True):
        adj = gumbel_sigmoid(self.log_alpha, self.uniform, bs, tau=tau, hard=drawhard)
        return adj

    def get_proba(self):
        """Returns probability of getting one"""
        return torch.sigmoid(self.log_alpha)

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 5)

class BaseModel(nn.Module):
    def __init__(self, num_vars, num_layers, hid_dim, num_params, nonlin="leaky-relu",
                 intervention=False, intervention_type="perfect",
                 intervention_knowledge="known", num_regimes=1):
        """
        :param int num_vars: number of variables in the system
        :param int num_layers: number of hidden layers
        :param int hid_dim: number of hidden units per layer
        :param int num_params: number of parameters per conditional *outputted by MLP*
        :param str nonlin: which nonlinearity to use
        :param boolean intervention: if True, use loss that take into account interventions
        :param str intervention_type: type of intervention: perfect or imperfect
        :param str intervention_knowledge: if False, don't use the intervention targets
        :param int num_regimes: total number of regimes
        """
        super().__init__()
        self.d = num_vars
        self.num_vars = num_vars
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.num_params = num_params
        self.nonlin = nonlin
        self.gumbel = True
        self.intervention = intervention
        self.intervention_type = intervention_type
        self.intervention_knowledge = intervention_knowledge
        self.num_regimes = num_regimes

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        # Those parameter might be learnable, but they do not depend on parents.
        self.extra_params = []

        if not(not self.intervention or \
        (self.intervention and self.intervention_type == "perfect" and self.intervention_knowledge == "known") or \
        (self.intervention and self.intervention_type == "perfect" and self.intervention_knowledge == "unknown") or \
        (self.intervention and self.intervention_type == "imperfect" and self.intervention_knowledge == "known")):
            raise ValueError("Not implemented")

        if not self.intervention:
            print("No intervention")
            self.intervention_type = "perfect"
            self.intervention_knowledge = "known"

        # initialize current adjacency matrix
        self.adjacency = torch.ones((self.num_vars, self.num_vars)) - torch.eye(self.num_vars)
        self.gumbel_adjacency = GumbelAdjacency(self.num_vars)

        self.zero_weights_ratio = 0.
        self.numel_weights = 0

        # Instantiate the parameters of each layer in the model of each variable
        for i in range(self.num_layers + 1):
            in_dim = self.hid_dim
            out_dim = self.hid_dim

            # first layer
            if i == 0:
                in_dim = self.num_vars

            # last layer
            if i == self.num_layers:
                out_dim = self.num_params

            # if interv are imperfect or unknown, generate 'num_regimes' MLPs per conditional
            if self.intervention and (self.intervention_type == 'imperfect' or
                                      self.intervention_knowledge == 'unknown'):
                self.weights.append(nn.Parameter(torch.zeros(self.num_vars,
                                                             out_dim, in_dim,
                                                             self.num_regimes)))
                self.biases.append(nn.Parameter(torch.zeros(self.num_vars, out_dim,
                                                            self.num_regimes)))
                self.numel_weights += self.num_vars * out_dim * in_dim * self.num_regimes
            # for perfect interv, generate only one MLP per conditional
            elif not self.intervention or self.intervention_type == 'perfect':
                self.weights.append(nn.Parameter(torch.zeros(self.num_vars, out_dim, in_dim)))
                self.biases.append(nn.Parameter(torch.zeros(self.num_vars, out_dim)))
                self.numel_weights += self.num_vars * out_dim * in_dim
            else:
                if self.intervention_type not in ['perfect', 'imperfect']:
                    raise ValueError(f'{intervention_type} is not a valid for intervention type')
                if self.intervention_knowledge not in ['known', 'unknown']:
                    raise ValueError(f'{intervention_knowledge} is not a valid value for intervention knowledge')
                
    def compute_penalty(self, list_, p=2, target=0.):
        penalty = 0
        for m in list_:
            penalty += torch.norm(m - target, p=p) ** p
        return penalty

    def forward_given_params(self, x, weights, biases, mask=None, regime=None):
        """
        :param x: batch_size x num_vars
        :param weights: list of lists. ith list contains weights for ith MLP
        :param biases: list of lists. ith list contains biases for ith MLP
        :param mask: tensor, batch_size x num_vars
        :param regime: np.ndarray, shape=(batch_size,)
        :return: batch_size x num_vars * num_params, the parameters of each variable conditional
        """
        bs = x.size(0)
        num_zero_weights = 0

        for layer in range(self.num_layers + 1):
            # First layer, apply the mask
            if layer == 0:
                # sample the matrix M that will be applied as a mask at the MLP input
                M = self.gumbel_adjacency(bs)
                adj = self.adjacency.unsqueeze(0)

                if not self.intervention:
                    x = torch.einsum("tij,bjt,ljt,bj->bti", weights[layer], M, adj, x) + biases[layer]
                elif self.intervention_type == "perfect" and self.intervention_knowledge == "known":
                    # the mask is not applied here, it is applied in the loss term
                    x = torch.einsum("tij,bjt,ljt,bj->bti", weights[layer], M, adj, x) + biases[layer]
                else:
                    assert mask is not None, 'Mask is not set!'
                    assert regime is not None, 'Regime is not set!'

                    regime = torch.from_numpy(regime)
                    R = mask

                    if self.intervention_knowledge == "unknown":
                        # sample the matrix R and totally mask the
                        # input of MLPs that are intervened on (in R)
                        self.interv_w = self.gumbel_interv_w(bs, regime)
                        R = self.interv_w
                        M = torch.einsum("bjt,bt->bjt", M, R)

                    # transform the mask format from bs x num_vars
                    # to bs x num_vars x num_regimes, in order to select the
                    # MLP parameter corresponding to the regime
                    R = (1 - R).type(torch.int64)
                    R = R * regime.unsqueeze(1)
                    R = torch.zeros(R.size(0), self.num_vars, self.num_regimes).scatter_(2, R.unsqueeze(2), 1)

                    # apply the first MLP layer with the mask M and the
                    # parameters 'selected' by R
                    w = torch.einsum('tijk, btk -> btij', weights[layer], R)
                    x = torch.einsum("btij, bjt, ljt, bj -> bti", w, M, adj, x)
                    x += torch.einsum("btk,tik->bti", R, biases[layer])

            # 2nd layer and more
            else:
                if self.intervention and (self.intervention_type == "imperfect" or self.intervention_knowledge == "unknown"):
                    w = torch.einsum('tijk, btk -> btij', weights[layer], R)
                    x = torch.einsum("btij, btj -> bti", w, x)
                    x += torch.einsum("btk,tik->bti", R, biases[layer])
                else:
                    x = torch.einsum("tij,btj->bti", weights[layer], x) + biases[layer]

            # count number of zeros
            num_zero_weights += weights[layer].numel() - weights[layer].nonzero().size(0)

            # apply non-linearity
            if layer != self.num_layers:
                x = F.leaky_relu(x) if self.nonlin == "leaky-relu" else torch.sigmoid(x)

        self.zero_weights_ratio = num_zero_weights / float(self.numel_weights)

        return torch.unbind(x, 1)

    def adj(self):
        """Get weighted adjacency matrix"""
        return self.gumbel_adjacency.get_proba() * self.adjacency

    def reset_params(self):
        with torch.no_grad():
            for node in range(self.num_vars):
                for i, w in enumerate(self.weights):
                    w = w[node]
                    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('leaky_relu'))
                for i, b in enumerate(self.biases):
                    b = b[node]
                    b.zero_()

    def get_parameters(self):
        """
        Will get only parameters with requires_grad == True
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: corresponding dicts of parameters
        """
        params = []

        weights = []
        for w in self.weights:
            weights.append(w)
        params.append(weights)

        biases = []
        for b in self.biases:
            biases.append(b)
        params.append(biases)

        return tuple(params)

    def get_distribution(self, density_params):
        raise NotImplementedError


class FlowModel(BaseModel):
    """
    Abstract class for normalizing flow model
    """
    def __init__(self, num_vars, num_layers, hid_dim, num_params, nonlin="leaky-relu",
                 intervention=False, intervention_type="perfect",
                 intervention_knowledge="known", num_regimes=1):
        super().__init__(num_vars, num_layers, hid_dim, num_params, nonlin=nonlin,
                         intervention=intervention,
                         intervention_type=intervention_type,
                         intervention_knowledge=intervention_knowledge,
                         num_regimes=num_regimes)
        self.reset_params()

    def compute_log_likelihood(self, x, weights, biases, detach=False, mask=None, regime=None):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :param weights: list of tensor that are coherent with self.weights
        :param biases: list of tensor that are coherent with self.biases
        :param mask: tensor, shape=(batch_size, num_vars)
        :param regime: np.ndarray, shape=(batch_size,)
        :return: (batch_size, num_vars) log-likelihoods
        """
        density_params = self.forward_given_params(x, weights, biases, mask, regime)
        return self._log_likelihood(x, density_params)

    def reset_params(self):
        super().reset_params()
        if "flow" in self.__dict__ and hasattr(self.flow, "reset_parameters"):
            self.flow.reset_parameters()


class DeepSigmoidalFlowModel(FlowModel):
    def __init__(self, num_vars, cond_n_layers, cond_hid_dim, cond_nonlin, flow_n_layers, flow_hid_dim,
                 intervention=False, intervention_type="perfect",
                 intervention_knowledge="known", num_regimes=1):
        """
        Deep Sigmoidal Flow model

        :param int num_vars: number of variables
        :param int cond_n_layers: number of layers in the conditioner
        :param int cond_hid_dim: number of hidden units in the layers of the conditioner
        :param str cond_nonlin: type of non-linearity used in the conditioner
        :param int flow_n_layers: number of DSF layers
        :param int flow_hid_dim: number of hidden units in the DSF layers
        :param boolean intervention: True if use interventional version (DCDI)
        :param str intervention_type: Either perfect or imperfect
        :param str intervention_knowledge: Either known or unkown
        :param int num_regimes: total number of regimes in the data
        """
        flow_n_conditioned = flow_hid_dim

        # Conditioner model initialization
        n_conditioned_params = flow_n_conditioned * 3 * flow_n_layers  # Number of conditional params for each variable
        super().__init__(num_vars, cond_n_layers, cond_hid_dim, num_params=n_conditioned_params, nonlin=cond_nonlin,
                         intervention=intervention,
                         intervention_type=intervention_type,
                         intervention_knowledge=intervention_knowledge,
                         num_regimes=num_regimes)
        self.cond_n_layers = cond_n_layers
        self.cond_hid_dim = cond_hid_dim
        self.cond_nonlin = cond_nonlin

        # Flow model initialization
        self.flow_n_layers = flow_n_layers
        self.flow_hid_dim = flow_hid_dim
        self.flow_n_params_per_var = flow_hid_dim * 3 * flow_n_layers  # total number of params
        self.flow_n_cond_params_per_var = n_conditioned_params  # number of conditional params
        self.flow_n_params_per_layer = flow_hid_dim * 3  # number of params in each flow layer
        self.flow = SigmoidFlow(flow_hid_dim)

        # Shared density parameters (i.e, those that are not produced by the conditioner)
        self.shared_density_params = torch.nn.Parameter(torch.zeros(self.flow_n_params_per_var -
                                                                    self.flow_n_cond_params_per_var))

    def reset_params(self):
        super().reset_params()
        if "flow" in self.__dict__:
            self.flow.reset_parameters()
        if "shared_density_params" in self.__dict__:
            self.shared_density_params.data.uniform_(-0.001, 0.001)

    def _log_likelihood(self, x, density_params):
        """
        Compute the log likelihood of x given some density specification.

        :param x: torch.Tensor, shape=(batch_size, num_vars), the input for which to compute the likelihood.
        :param density_params: tuple of torch.Tensor, len=n_vars, shape of elements=(batch_size, n_flow_params_per_var)
            The parameters of the DSF model that were produced by the conditioner.
        :return: pseudo joint log-likelihood
        """
        # Convert the shape to (batch_size, n_vars, n_flow_params_per_var)
        density_params = torch.cat([x[None, :, :] for x in density_params], dim=0).transpose(0, 1)
        assert len(density_params.shape) == 3
        assert density_params.shape[0] == x.shape[0]
        assert density_params.shape[1] == self.num_vars
        assert density_params.shape[2] == self.flow_n_cond_params_per_var

        # Inject shared parameters here
        # Add the shared density parameters in each layer's parameter vectors
        # The shared parameters are different for each layer
        # All batch elements receive the same shared parameters
        conditional = density_params.view(density_params.shape[0], density_params.shape[1], self.flow_n_layers, 3, -1)
        shared = \
            self.shared_density_params.view(self.flow_n_layers, 3, -1)[None, None, :, :, :].repeat(conditional.shape[0],
                                                                                                   conditional.shape[1],
                                                                                                   1, 1, 1)
        density_params = torch.cat((conditional, shared), -1).view(conditional.shape[0], conditional.shape[1], -1)
        assert density_params.shape[2] == self.flow_n_params_per_var

        logdet = Variable(torch.zeros((x.shape[0], self.num_vars)))
        h = x.view(x.size(0), -1)
        for i in range(self.flow_n_layers):
            # Extract params of the current flow layer. Shape is (batch_size, n_vars, self.flow_n_params_per_layer)
            params = density_params[:, :, i * self.flow_n_params_per_layer: (i + 1) * self.flow_n_params_per_layer]
            h, logdet = self.flow(h, logdet, params)

        assert x.shape[0] == h.shape[0]
        assert x.shape[1] == h.shape[1]
        zeros = Variable(torch.zeros(x.shape[0], self.num_vars))
        # Not the joint NLL until we have a DAG
        pseudo_joint_nll = - log_normal(h, zeros, zeros + 1.0) - logdet

        # We return the log product (averaged) of conditionals instead of the logp for each conditional.
        #      Shape is (batch x 1) instead of (batch x n_vars).
        return - pseudo_joint_nll
