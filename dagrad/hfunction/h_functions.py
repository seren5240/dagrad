import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
import torch
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
            pass
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
            I = torch.eye(d, device=device)
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
