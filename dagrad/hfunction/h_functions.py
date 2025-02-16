import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
import torch
import torch.nn as nn
import scipy


def normalize(v):
    return v / torch.linalg.vector_norm(v)


class SCCPowerIteration(nn.Module):
    def __init__(self, init_adj_mtx, d, update_scc_freq=1000):
        super().__init__()
        self.d = d
        self.update_scc_freq = update_scc_freq

        self._dummy_param = nn.Parameter(
            torch.zeros(1), requires_grad=False
        )  # Used to track device

        self.scc_list = None
        self.update_scc(init_adj_mtx)

        self.register_buffer("v", None)
        self.register_buffer("vt", None)
        self.initialize_eigenvectors(init_adj_mtx)

        self.n_updates = 0

    @property
    def device(self):
        return self._dummy_param.device

    def initialize_eigenvectors(self, adj_mtx):
        self.v, self.vt = torch.ones(size=(2, self.d), device=self.device, dtype=torch.double)
        self.v = normalize(self.v)
        self.vt = normalize(self.vt)
        return self.power_iteration(adj_mtx, 5)

    def update_scc(self, adj_mtx):
        n_components, labels = scipy.sparse.csgraph.connected_components(
            csgraph=scipy.sparse.coo_matrix(adj_mtx.cpu().detach().numpy()),
            directed=True,
            return_labels=True,
            connection="strong",
        )
        self.scc_list = []
        for i in range(n_components):
            scc = np.where(labels == i)[0]
            self.scc_list.append(scc)
        # print(len(self.scc_list))

    def power_iteration(self, adj_mtx, n_iter=5):
        matrix = adj_mtx**2
        for scc in self.scc_list:
            if len(scc) == self.d:
                sub_matrix = matrix
                v = self.v
                vt = self.vt
                for i in range(n_iter):
                    v = normalize(sub_matrix.mv(v) + 1e-6 * v.sum())
                    vt = normalize(sub_matrix.T.mv(vt) + 1e-6 * vt.sum())
                self.v = v
                self.vt = vt

            else:
                sub_matrix = matrix[scc][:, scc]
                v = self.v[scc]
                vt = self.vt[scc]
                for i in range(n_iter):
                    v = normalize(sub_matrix.mv(v) + 1e-6 * v.sum())
                    vt = normalize(sub_matrix.T.mv(vt) + 1e-6 * vt.sum())
                self.v[scc] = v
                self.vt[scc] = vt

        return matrix

    def compute_gradient(self, adj_mtx):
        if self.n_updates % self.update_scc_freq == 0:
            self.update_scc(adj_mtx)
            self.initialize_eigenvectors(adj_mtx)

        # matrix = self.power_iteration(4)
        matrix = self.initialize_eigenvectors(adj_mtx)

        gradient = torch.zeros(size=(self.d, self.d), device=self.device, dtype=torch.double)
        for scc in self.scc_list:
            if len(scc) == self.d:
                v = self.v
                vt = self.vt
                gradient = torch.outer(vt, v) / torch.inner(vt, v)
            else:
                v = self.v[scc]
                vt = self.vt[scc]
                gradient[scc][:, scc] = torch.outer(vt, v) / torch.inner(vt, v)

        gradient += 100 * torch.eye(self.d, device=self.device)
        # gradient += matrix.T

        self.n_updates += 1

        return gradient, matrix

class h_fn:
    @staticmethod
    def user_h(W,**kwargs):
        """
        This is a user-defined h function. Users can define their own h function by customizing this function.

        Parameters
        ----------
        W : numpy array or torch tensor
            Weight matrix.

        Returns
        -------
        h : float
            Value of the h function.
        G_h : numpy array or torch tensor
            Gradient of the h function.

        Important
        ---------
        If users work with **numpy** as the computation library, the output of the ``user_h`` function should be 
        (value of h, gradient of h) in numpy arrays.
        If users work with **torch** as the computation library, the output of the ``user_h`` function should be 
        the value of h in a torch tensor.
        """
        if isinstance(W,np.ndarray):
            pass
        elif isinstance(W,torch.Tensor):
            # A = W ** 2
            # h = -torch.slogdet(torch.eye(W.shape[0]) - A)[1]
            # return h
            user_params = kwargs.get('user_params', None)
            is_prescreen = user_params.get('is_prescreen')
            if is_prescreen:
                return torch.tensor(0.0, dtype=W.dtype, device=W.device)
            power_grad: SCCPowerIteration = user_params.get('power_grad')
            grad, A = power_grad.compute_gradient(W)
            h_val = (grad.detach() * A).sum()
            return h_val
        else:
            raise ValueError("W must be either numpy array or torch tensor")

        raise NotImplementedError("User defined h function is not implemented yet. User are free to define their own h function.")

    @staticmethod
    def h_exp_sq(W, **kwargs):

        d = W.shape[0]
        if isinstance(W, np.ndarray):
            E = sla.expm(W * W)  # (Zheng et al. 2018)
            h = np.trace(E) - d
            G_h = E.T * W * 2
            return h, G_h
        elif isinstance(W, torch.Tensor):
            E = torch.linalg.matrix_exp(W * W)
            h = torch.trace(E) - d
            return h
        else:
            raise ValueError("W must be either numpy array or torch tensor")
        
    @staticmethod
    def h_logdet_sq(W, **kwargs) :
        
        s = kwargs.get('s', 1.0)
        d = W.shape[0]
        if isinstance(W,np.ndarray):
            I = np.eye(d)
            M = s* I - W * W
            h = - la.slogdet(M)[1] + d * np.log(s)
            G_h = 2 * W * sla.inv(M).T 
            return h, G_h
        elif isinstance(W, torch.Tensor):
            device = W.device
            dtype = W.dtype
            I = torch.eye(d,dtype=dtype, device=device)
            s = torch.tensor(s, device=device, dtype=dtype)
            M = s* I - W * W
            h = - torch.slogdet(M)[1] + d * torch.log(s)
            return h
        else:
            raise ValueError("W must be either numpy array or torch tensor")


        
    @staticmethod
    def h_poly_sq(W, **kwargs):
        
        d = W.shape[0]
        if isinstance(W, np.ndarray):
            M = np.eye(d) + W * W / d
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            G_h = E.T * W * 2
            return h, G_h
        elif isinstance(W, torch.Tensor):
            device = W.device
            dtype = W.dtype
            M = torch.eye(d,device=device,dtype = dtype) + W * W / d
            E = torch.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            return h
        
    @staticmethod
    def h_exp_abs(W, **kwargs):
        
        d = W.shape[0]
        if isinstance(W, np.ndarray):
            E = sla.expm(np.abs(W))
            h = np.trace(E) - d
            G_h = np.sign(W) * E.T
            return h, G_h
        elif isinstance(W, torch.Tensor):
            E = torch.linalg.matrix_exp(torch.abs(W))
            h = torch.trace(E) - d
            return h
        else:
            raise ValueError("W must be either numpy array or torch tensor")
        
    @staticmethod
    def h_poly_abs(W, **kwargs):
       
        d = W.shape[0]
        if isinstance(W, np.ndarray):
            M = np.eye(d) + np.abs(W) / d
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            G_h = np.sign(W) * E.T
            return h, G_h
        else:
            device = W.device
            dtype = W.dtype
            M = torch.eye(d,device=device,dtype =dtype) + torch.abs(W) / d
            E = torch.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            return h
    @staticmethod
    def h_logdet_abs(W, **kwargs):
        
        s = kwargs.get('s', 1.0)
        d = W.shape[0]
        if isinstance(W,np.ndarray):
            I = np.eye(d)
            M = s* I - np.abs(W)
            h = - la.slogdet(M)[1] + d * np.log(s)
            G_h = np.sign(W) * sla.inv(M).T 
            return h, G_h
        elif isinstance(W, torch.Tensor):
            device = W.device
            dtype = W.dtype
            I = torch.eye(d, device=device)
            s = torch.tensor(s, device=device, dtype=dtype)
            M = s* I - torch.abs(W)
            h = - torch.slogdet(M)[1] + d * torch.log(s)
            return h
        else:
            raise ValueError("W must be either numpy array or torch tensor")


    #####################################################
    # h function for TOPO is different from the above
    # TOPO uses [\nabla h(W)]_{W = |W|}, evaluate the gradient of h(W) at W = |W|
    # logdet: h(W) = -log det(sI-W) + d log (s), \nabla h(W) =  (sI-W)^{-T}
    # exp: h(W) = Tr(exp(W)) - d, \nabla h(W) = exp(W).T
    # poly: h(W) = Tr(I+W/d)^d-d, \nabla h(W) = (I+W/d)^{d-1}.T
    # TOPO uses the absolute value of W, so the gradient is the same as the above
    # without multiplication of np.sign(W)
    #####################################################

    @staticmethod
    def h_exp_topo(W, **kwargs):
        """
        Evaluate value and gradient of acyclicity constraint.
        """
        d = W.shape[0]
        E = sla.expm(np.abs(W))  # (Zheng et al. 2018)
        h = np.trace(E) - d
        G_h =  E.T
        return h, G_h
    @staticmethod
    def h_poly_topo(W, **kwargs):
        """
        Evaluate value and gradient of acyclicity constraint.
        """
        d = W.shape[0]
        M = np.eye(d) + np.abs(W) / d
        E = np.linalg.matrix_power(M, d - 1)
        h = (E.T * M).sum() - d
        G_h =  E.T
        return h, G_h
    @staticmethod
    def h_logdet_topo(W, **kwargs):
        """
        Evaluate value and gradient of acyclicity constraint.
        """
        s = kwargs.get('s', 1.0)
        d = W.shape[0]
        I = np.eye(d)
        M = s* I - np.abs(W)
        h = - la.slogdet(M)[1] + d * np.log(s)
        G_h =  sla.inv(M).T 
        return h, G_h
