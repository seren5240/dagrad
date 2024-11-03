import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from ..utils.topo_utils import create_Z,create_new_topo

class NotearsMLP(nn.Module):
    def __init__(self, 
                 dims, 
                  activation = 'sigmoid', 
                  bias = True,
                  dtype = torch.float64) -> None:
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2 and dims[-1] == 1, "Invalid dimension size or output dimension."
        self.d = dims[0]
        self.d1 = dims[1]
        self.dims = dims
        self.depth = len(dims)
        self.bias = bias
        self.dtype = dtype
        self.layerlist = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        self._create_layerlist()
        self.init_layer0_weights_and_biases()
        
        # with torch.no_grad():
        #     for i, layer in enumerate(self.layerlist[0]):
        #         layer.weight[:,i] = 0.0
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError("Invalid activation function.")
        
        for i, layer in enumerate(self.layerlist[0]):
            layer.weight.register_hook(self.make_hook_function(i))

        
    @torch.no_grad()
    def init_layer0_weights_and_biases(self):    
        for i,layer in enumerate(self.layerlist[0]):
            nn.init.zeros_(layer.weight[:,i])
            if self.bias:
                nn.init.zeros_(layer.bias)

    def _create_layerlist(self):

        for i in range(0, self.depth - 1):
            layers = nn.ModuleList([nn.Linear(self.dims[i], self.dims[i + 1], bias=self.bias, dtype= self.dtype) for _ in range(self.d)])
            self.layerlist.append(layers)

    @staticmethod
    def make_hook_function(column_idx):
        def hook_function(grad):
            grad_clone = grad.clone()
            grad_clone[:, column_idx] = 0.0  # Zero out the specific column's gradient
            return grad_clone
        return hook_function
    
    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        for layer in self.layerlist:
            for l in layer:
                reg += torch.sum(l.weight ** 2)
                if self.bias:
                    reg += torch.sum(l.bias ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = 0.
        for l in self.layerlist[0]:
            reg+=torch.sum(torch.abs(l.weight))
        return reg

    def forward(self,x):
        output = []
        binary = torch.all((x == 0) | (x == 1))
        for i in range(self.d):
            x_copy = x.clone()
            for j in range(self.depth-1): # [d, 40, 1] depth = 3, j = 0,1
                x_copy = self.layerlist[j][i](x_copy)
                if j < self.depth - 2:
                    x_copy = self.activation(x_copy)
            if binary:
                x_copy = self.sigmoid(x_copy)
            output.append(x_copy)
        return torch.cat(output, dim=1)
    
    # def h_func(self):
    #     A = torch.zeros(self.d,self.d)
    #     for i in range(self.d):
    #         A[:,i] = torch.sum(self.layerlist[0][i].weight**2,dim=0)
    #     h = torch.trace(torch.linalg.matrix_exp(A)) - self.d
    #     return h

    def adj(self):
        A = torch.zeros(self.d,self.d, dtype = self.dtype)
        for i in range(self.d):
            A[:,i] = torch.sum(self.layerlist[0][i].weight**2,dim=0)
        W = torch.sqrt(A)
        return W
    
    @torch.no_grad()
    def fc1_to_adj(self):
        A = torch.zeros(self.d,self.d,dtype=self.dtype)
        for i in range(self.d):
            A[:,i] = torch.sum(self.layerlist[0][i].weight**2,dim=0)
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()
        return W
    


class DagmaMLP(nn.Module):
    def __init__(self, dims, activation = 'sigmoid', bias = True, dtype = torch.float64) -> None:
        super(DagmaMLP, self).__init__()
        assert len(dims) >= 2 and dims[-1] == 1, "Invalid dimension size or output dimension."
        self.d = dims[0]
        self.d1 = dims[1]
        self.dims = dims
        self.depth = len(dims)
        self.bias = bias
        self.dtype = dtype
        self.sigmoid = nn.Sigmoid()
        self.layerlist = nn.ModuleList()
        self._create_layerlist()
        self.init_layer0_weights_and_biases()
        # with torch.no_grad():
        #     for i, layer in enumerate(self.layerlist[0]):
        #         layer.weight[:,i] = 0.0
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError("Invalid activation function.")
        
        for i, layer in enumerate(self.layerlist[0]):
            layer.weight.register_hook(self.make_hook_function(i))

        
    @torch.no_grad()
    def init_layer0_weights_and_biases(self):    
        for layer in self.layerlist[0]:
            nn.init.zeros_(layer.weight)
            if self.bias:
                nn.init.zeros_(layer.bias)

    def _create_layerlist(self):
        for i in range(0, self.depth - 1):
            layers = nn.ModuleList([nn.Linear(self.dims[i], self.dims[i + 1], bias=self.bias, dtype=self.dtype) for _ in range(self.d)])
            self.layerlist.append(layers)

    @staticmethod
    def make_hook_function(column_idx):
        def hook_function(grad):
            grad_clone = grad.clone()
            grad_clone[:, column_idx] = 0.0  # Zero out the specific column's gradient
            return grad_clone
        return hook_function
    
    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        for layer in self.layerlist:
            for l in layer:
                reg += torch.sum(l.weight ** 2)
                if self.bias:
                    reg += torch.sum(l.bias ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = 0.
        for l in self.layerlist[0]:
            reg+=torch.sum(torch.abs(l.weight))
        return reg

    def forward(self,x):
        output = []
        binary = torch.all((x == 0) | (x == 1))
        for i in range(self.d):
            x_copy = x.clone()
            for j in range(self.depth-1):
                x_copy = self.layerlist[j][i](x_copy)
                if j < self.depth - 2:
                    x_copy = self.activation(x_copy)
            if binary:
                x_copy = self.sigmoid(x_copy)
            output.append(x_copy)
        return torch.cat(output, dim=1)
    
    # def h_func(self):
    #     A = torch.zeros(self.d,self.d)
    #     for i in range(self.d):
    #         A[:,i] = torch.sum(self.layerlist[0][i].weight**2,dim=0)
    #     h = torch.trace(torch.linalg.matrix_exp(A)) - self.d
    #     return h

    # def adj(self):
    #     A = torch.zeros(self.d,self.d)
    #     for i in range(self.d):
    #         A[:,i] = torch.sum(self.layerlist[0][i].weight**2,dim=0)
    #     # mask = 1 - torch.eye(self.d, self.d)
    #     # epsilon = 1e-8
    #     # zero_off_diag_mask = (A == 0) & mask.bool()
    #     # A[zero_off_diag_mask] += epsilon
    #     epsilon = 1e-8
    #     if torch.all(A<epsilon):
    #         W = torch.sqrt(A+epsilon)
    #     else:
    #         W = torch.sqrt(A)
    #     return W
    
    def adj(self):
        A = torch.zeros(self.d,self.d,dtype=self.dtype)
        for i in range(self.d):
            # A[:,i] = torch.sum(torch.abs(self.layerlist[0][i].weight),dim=0)
            A[:,i] = torch.sum(self.layerlist[0][i].weight**2,dim=0)
        return A
    
    @torch.no_grad()
    def fc1_to_adj(self):
        A = torch.zeros(self.d,self.d,dtype=self.dtype)
        for i in range(self.d):
            A[:,i] = torch.sum(self.layerlist[0][i].weight**2,dim=0)
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()
        return W



class TopoMLP(nn.Module):

    def __init__(self, dims, activation = 'sigmoid', bias = True, dtype = torch.float64):
        super(TopoMLP, self).__init__()
        assert len(dims) >= 2 and dims[-1] == 1, "Invalid dimension size or output dimension."
        self.d = dims[0]
        self.d1 = dims[1]
        self.dtype = dtype
        self.dims = dims
        self.depth = len(dims)
        self.sigmoid = nn.Sigmoid()
        self.layerlist = nn.ModuleList()
        self.bias = bias
        self._create_layerlist()
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError("Invalid activation function.")
        if self.bias == True:
            self.biasList = nn.ParameterList([nn.Parameter(torch.zeros(1,self.d1).to(dtype=self.dtype)) for _ in range(self.d)])

    def _create_layerlist(self):
        # Initialize the first layer separately due to its unique structure
        layer0 = nn.ModuleList([nn.Linear(1, self.d1, bias=False).to(dtype=self.dtype) for _ in range(self.d * self.d)])
        self.layerlist.append(layer0)
        # For subsequent layers, use a more streamlined approach
        for i in range(1, self.depth - 1):
            layers = nn.ModuleList([nn.Linear(self.dims[i], self.dims[i + 1], bias=self.bias).to(dtype=self.dtype) for _ in range(self.d)])
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
        binary = torch.all((x == 0) | (x == 1))
        layer0_weights = torch.cat([self.layerlist[0][ll].weight for ll in range(self.d * ith, self.d * (ith + 1))],
                                       dim=1).T
        if self.bias:
            x = torch.mm(x, layer0_weights)+self.biasList[ith]
        else:
            x = torch.mm(x, layer0_weights)
        for ii in range(1, self.depth - 1):
            x = self.activation(x)
            # x = F.sigmoid(x)  # Consider using F.relu(x) for ReLU activation
            x = self.layerlist[ii][ith](x)
        if binary:
            x = self.sigmoid(x)
        return x

    #

    def forward(self, x):  # [n,d] ->[n,d]
        x = x.to(dtype=self.dtype)
        output = [self._forward_i(x, ii) for ii in range(self.d)]
        return torch.cat(output, dim=1)


    @torch.no_grad()
    def freeze_grad_f_i(self,ith):
        # freeze all the gradient of all the parameters related to f_i
        for k in range(self.d):
            self.layerlist[0][int(k+self.d*ith)].weight.requires_grad = False
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

        self.reset_by_topo(topo = topo0)
        freeze_idx = [oo for oo in range(self.d) if oo not in topo0[wherej:(wherei + 1)]]
        if freeze_idx:
            for ith in freeze_idx:
                self.freeze_grad_f_i(ith)

    def layer0_l1_reg(self):
        return sum(torch.sum(torch.abs(vec.weight)) for vec in self.layerlist[0])

    def l2_reg(self):
        if self.bias:
            return sum(torch.sum(vec.weight ** 2) for layer in self.layerlist for vec in layer)+ sum(torch.sum(vec ** 2) for vec in self.biasList)
        else:
            return sum(torch.sum(vec.weight ** 2) for layer in self.layerlist for vec in layer)
    
    def adj(self):
        W = torch.zeros((self.d * self.d), dtype = self.dtype)
        for count, vec in enumerate(self.layerlist[0]):
            # W[count] = torch.sqrt(torch.sum(vec.weight ** 2))
            W[count] = torch.norm(vec.weight.data, p=2)
        W = torch.reshape(W, (self.d, self.d)).t()

        return W 


    @torch.no_grad()
    def get_gradient_F(self):
        G_grad = torch.zeros(self.d ** 2, dtype= self.dtype)
        for count, vec in enumerate(self.layer0):
            G_grad[count] = torch.norm(vec.weight.grad, p=2)
        G_grad = torch.reshape(G_grad, (self.d, self.d)).t()
        return G_grad.numpy()

    @torch.no_grad()
    def layer0_to_adj(self):

        W = torch.zeros((self.d * self.d), dtype = self.dtype)
        for count, vec in enumerate(self.layerlist[0]):
            # W[count] = torch.sqrt(torch.sum(vec.weight ** 2))
            W[count] = torch.norm(vec.weight.data, p=2)
        W = torch.reshape(W, (self.d, self.d)).t()

        return W.numpy()
    
