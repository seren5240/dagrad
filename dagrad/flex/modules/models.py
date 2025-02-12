import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
            torch.Tensor(num_linear, input_features, output_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
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
        self, dims, activation="sigmoid", bias=True, dtype=torch.float64
    ) -> None:
        torch.set_default_dtype(dtype)
        super().__init__()
        assert (
            len(dims) >= 2 and dims[-1] == 1
        ), "Invalid dimension size or output dimension."
        self.d = dims[0]
        self.dims = dims
        self.bias = bias
        self.layers = nn.ModuleList()

        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias, dtype=dtype)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        if activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError("Activation function not supported.")
        self.fc2 = nn.ModuleList()
        for k in range(len(dims) - 2):
            self.fc2.append(
                LocallyConnected(self.d, dims[k + 1], dims[k + 2], bias=bias)
            )

        self.fc1.weight.register_hook(self.make_hook_function(self.d))

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

    def l1_loss(self):
        """Take l1 norm of fc1 weight"""
        return torch.sum(torch.abs(self.fc1.weight))

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = self.activation(x)
            x = fc(x)
        x = x.squeeze(dim=2)
        return x

    def adj(self):
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight**2, dim=1).t()
        return A

    @torch.no_grad()
    def fc1_to_adj(self):
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight**2, dim=1).t()
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W
    
    def forward_given_params(self, x, weights, biases):
        """

        :param x: batch_size x num_vars
        :param weights: list of lists. ith list contains weights for ith MLP
        :param biases: list of lists. ith list contains biases for ith MLP
        :return: batch_size x num_vars * num_params, the parameters of each variable conditional
        """
        # num_zero_weights = 0
        num_layers = len(self.layers)
        for k in range(num_layers + 1):
            # apply affine operator
            if k == 0:
                adj = self.adj().unsqueeze(0)
                # print(f'dimensions of weights[k] is {weights[k].shape} and of adj is {adj.shape} and of x is {x.shape}')
                x = torch.einsum("tij,ljt,bj->bti", weights[k], adj, x) + biases[k]
            else:
                x = torch.einsum("tij,btj->bti", weights[k], x) + biases[k]

            # count num of zeros
            # num_zero_weights += weights[k].numel() - weights[k].nonzero().size(0)

            # apply non-linearity
            if k != num_layers:
                x = F.leaky_relu(x) # if self.nonlin == "leaky-relu" else torch.sigmoid(x)

        return torch.unbind(x, 1)

    def get_parameters(self):
        params = []
        
        weights = []
        for w in self.fc2:
            weights.append(w.weight)
        params.append(weights)

        biases = []
        for j, b in enumerate(self.fc2):
            biases.append(b.bias)
        params.append(biases)

        return tuple(params)

    def get_distribution(self, dp):
        return torch.distributions.normal.Normal(dp[0], torch.exp(dp[1]))
    
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

        # if len(extra_params) != 0:
        #     extra_params = self.transform_extra_params(self.extra_params)
        log_probs = []
        for i in range(self.d):
            density_param = list(torch.unbind(density_params[i], 1))
            # if len(extra_params) != 0:
                # density_param.extend(list(torch.unbind(extra_params[i], 0)))
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
