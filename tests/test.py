import sys
import os
from dagrad.core import dagrad
import unittest
import numpy as np
import time
from dagrad.utils import utils


# Test class for the NOTEARS algorithm
class TestNOTEARS(unittest.TestCase):
    def setUp(self):
        # setup for linear model
        self.n_linear = 1000
        self.d_linear = 5
        self.s0_linear = 10
        self.graph_type_linear = 'ER'
        self.sem_type_linear = 'gauss'

        # setup for logisitic model
        self.n_logistic = 10000
        self.d_logistic = 5
        self.s0_logistic = 10
        self.graph_type_logistic = 'ER'
        self.sem_type_logistic = 'logistic'

        # setup for nonlinear model
        self.n_nonlinear = 1000
        self.d_nonlinear = 5
        self.s0_nonlinear = 10
        self.graph_type_nonlinear = 'ER'
        self.sem_type_nonlinear = 'mlp'

        # general setup
        self.verbose = False



    
    # generate the data
    @staticmethod
    def generate_linear_data(n,d,s0,graph_type,sem_type,seed=111):
        utils.set_random_seed(seed=seed)
        B_true = utils.simulate_dag(d, s0, graph_type)
        W_true = utils.simulate_parameter(B_true)
        X = utils.simulate_linear_sem(W_true, n, sem_type)
        return X, W_true, B_true
    @staticmethod
    def generate_nonlinear_data(n,d,s0,graph_type,sem_type,seed=111):
        utils.set_random_seed(seed=seed)
        B_true = utils.simulate_dag(d, s0, graph_type)
        X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
        return X, B_true
    


    def testNotearsGauss1(self):
        
        # generate the data
        X, W_true, B_true = self.generate_linear_data(self.n_linear,self.d_linear,self.s0_linear,self.graph_type_linear,self.sem_type_linear)

        # print the simulation information
        print(f'\n\nWorking with linear model with __Numpy__ implementation: '
              f'\n\tSample size: {self.n_linear} '
              f'\n\tNumber of nodes: {self.d_linear}'
              f'\n\tNumber of edges: {self.s0_linear}'
              f'\n\tGraph type: {self.graph_type_linear}'
              f'\n\tSEM type: {self.sem_type_linear}\n')

        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'notears', method_options={'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for Notears (numpy): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testNotearsGauss2(self):

        X, W_true, B_true = self.generate_linear_data(self.n_linear,
                                                      self.d_linear,
                                                      self.s0_linear,
                                                      self.graph_type_linear,
                                                      self.sem_type_linear)
        # print the simulation information
        print(f'\n\nWorking with linear model with __Torch__ and __CPU__ implementation: '
              f'\n\tSample size: {self.n_linear} '
              f'\n\tNumber of nodes: {self.d_linear}'
              f'\n\tNumber of edges: {self.s0_linear}'
              f'\n\tGraph type: {self.graph_type_linear}'
              f'\n\tSEM type: {self.sem_type_linear}\n')

        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'notears', 
                               compute_lib = 'torch', device = 'cpu',method_options = {'verbose': self.verbose})
        time_end = time.time()

        print('Time taken for Notears (torch, cpu): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testNotearsGauss3(self):
        # generate the data
        X, W_true, B_true = self.generate_linear_data(self.n_linear,
                                                      self.d_linear,
                                                      self.s0_linear,
                                                      self.graph_type_linear,
                                                      self.sem_type_linear)
        # print the simulation information
        print(f'\n\nWorking with linear model with __Torch__ and __GPU__ implementation: '
              f'\n\tSample size: {self.n_linear} '
              f'\n\tNumber of nodes: {self.d_linear}'
              f'\n\tNumber of edges: {self.s0_linear}'
              f'\n\tGraph type: {self.graph_type_linear}'
              f'\n\tSEM type: {self.sem_type_linear}\n')

        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'notears',
                               compute_lib = 'torch', device = 'cuda',method_options = {'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for Notears (torch gpu): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testNotearsLogistic1(self):
        # generate the data
        X, W_true, B_true = self.generate_linear_data(self.n_logistic,
                                                      self.d_logistic,
                                                      self.s0_logistic,
                                                      self.graph_type_logistic,
                                                      self.sem_type_logistic)
        # print the simulation information
        print(f'\n\nWorking with linear model with __Numpy__ implementation: '
              f'\n\tSample size: {self.n_logistic} '
              f'\n\tNumber of nodes: {self.d_logistic}'
              f'\n\tNumber of edges: {self.s0_logistic}'
              f'\n\tGraph type: {self.graph_type_logistic}'
              f'\n\tSEM type: {self.sem_type_logistic}\n')
        


        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'notears', loss_fn = 'logistic', 
                             method_options={'verbose': self.verbose}, general_options={'lambda1':0.02})
        time_end = time.time()
        print('Time taken for Notears (numpy): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testNotearsLogistic2(self):
        X, W_true, B_true = self.generate_linear_data(self.n_logistic,
                                                      self.d_logistic,
                                                      self.s0_logistic,
                                                      self.graph_type_logistic,
                                                      self.sem_type_logistic)
        # print the simulation information
        print(f'\n\nWorking with linear model with __Torch__ and __CPU__ implementation: '
              f'\n\tSample size: {self.n_logistic} '
              f'\n\tNumber of nodes: {self.d_logistic}'
              f'\n\tNumber of edges: {self.s0_logistic}'
              f'\n\tGraph type: {self.graph_type_logistic}'
              f'\n\tSEM type: {self.sem_type_logistic}\n')
        
        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'notears', loss_fn = 'logistic',
                               compute_lib = 'torch', device = 'cpu', general_options={'lambda1':0.02},
                               method_options={'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for Notears torch cpu: ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)
    
    def testNotearsLogistic3(self):
        X, W_true, B_true = self.generate_linear_data(self.n_logistic,
                                                      self.d_logistic,
                                                      self.s0_logistic,
                                                      self.graph_type_logistic,
                                                      self.sem_type_logistic)
        # print the simulation information
        print(f'\n\nWorking with linear model with __Torch__ and __GPU__ implementation: '
              f'\n\tSample size: {self.n_logistic} '
              f'\n\tNumber of nodes: {self.d_logistic}'
              f'\n\tNumber of edges: {self.s0_logistic}'
              f'\n\tGraph type: {self.graph_type_logistic}'
              f'\n\tSEM type: {self.sem_type_logistic}\n')
        

        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'notears', loss_fn = 'logistic',
                               compute_lib = 'torch', device = 'cuda', general_options={'lambda1':0.02},
                               method_options = {'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for Notears (torch gpu): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)
        
    def testNotearsMLP1(self):
        X, B_true = self.generate_nonlinear_data(self.n_nonlinear,
                                                 self.d_nonlinear,
                                                 self.s0_nonlinear,
                                                 self.graph_type_nonlinear,
                                                 self.sem_type_nonlinear)

        print(f'\n\nWorking with nonlinear model with __torch__ and __CPU__ implementation: '
              f'\n\tSample size: {self.n_nonlinear} '
              f'\n\tNumber of nodes: {self.d_nonlinear}'
              f'\n\tNumber of edges: {self.s0_nonlinear}'
              f'\n\tGraph type: {self.graph_type_nonlinear}'
              f'\n\tSEM type: {self.sem_type_nonlinear}\n')

        time_start = time.time()
        W_est = dagrad(X = X, model = 'nonlinear', method = 'notears', method_options={'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for Notears torch cpu: ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)


    def testNotearsMLP2(self):
        X, B_true = self.generate_nonlinear_data(self.n_nonlinear,
                                                 self.d_nonlinear,
                                                 self.s0_nonlinear,
                                                 self.graph_type_nonlinear,
                                                 self.sem_type_nonlinear)

        print(f'\n\nWorking with nonlinear model with __torch__ and __GPU__ implementation: '
              f'\n\tSample size: {self.n_nonlinear} '
              f'\n\tNumber of nodes: {self.d_nonlinear}'
              f'\n\tNumber of edges: {self.s0_nonlinear}'
              f'\n\tGraph type: {self.graph_type_nonlinear}'
              f'\n\tSEM type: {self.sem_type_nonlinear}\n')

        time_start = time.time()
        W_est = dagrad(X = X, model = 'nonlinear', method = 'notears',
                               compute_lib = 'torch', device = 'cuda', method_options = {'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for Notears torch gpu: ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

# Test class for the DAGMA algorithm
class TestDAGMA(unittest.TestCase):
    def setUp(self):
        # setup for linear model
        self.n_linear = 1000
        self.d_linear = 5
        self.s0_linear = 10
        self.graph_type_linear = 'ER'
        self.sem_type_linear = 'gauss'

        # setup for logisitic model
        self.n_logistic = 10000
        self.d_logistic = 5
        self.s0_logistic = 10
        self.graph_type_logistic = 'ER'
        self.sem_type_logistic = 'logistic'

        # setup for nonlinear model
        self.n_nonlinear = 1000
        self.d_nonlinear = 5
        self.s0_nonlinear = 10
        self.graph_type_nonlinear = 'ER'
        self.sem_type_nonlinear = 'mlp'

        # general setup
        self.verbose = False

    @staticmethod
    def generate_linear_data(n,d,s0,graph_type,sem_type,seed=1234):
        utils.set_random_seed(seed=seed)
        B_true = utils.simulate_dag(d, s0, graph_type)
        W_true = utils.simulate_parameter(B_true)
        X = utils.simulate_linear_sem(W_true, n, sem_type)
        return X, W_true, B_true
    @staticmethod
    def generate_nonlinear_data(n,d,s0,graph_type,sem_type,seed=1234):
        utils.set_random_seed(seed=seed)
        B_true = utils.simulate_dag(d, s0, graph_type)
        X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
        return X, B_true
    
    def testDagmaGauss1(self):
        
        # generate the data
        X, W_true, B_true = self.generate_linear_data(self.n_linear,self.d_linear,self.s0_linear,self.graph_type_linear,self.sem_type_linear)

        # print the simulation information
        print(f'\n\nWorking with linear model with __Numpy__ implementation: '
              f'\n\tSample size: {self.n_linear} '
              f'\n\tNumber of nodes: {self.d_linear}'
              f'\n\tNumber of edges: {self.s0_linear}'
              f'\n\tGraph type: {self.graph_type_linear}'
              f'\n\tSEM type: {self.sem_type_linear}\n')
        
        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'dagma', method_options = {'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for dagma (numpy): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testDagmaGauss2(self):

        X, W_true, B_true = self.generate_linear_data(self.n_linear,
                                                      self.d_linear,
                                                      self.s0_linear,
                                                      self.graph_type_linear,
                                                      self.sem_type_linear)
        # print the simulation information
        print(f'\n\nWorking with linear model with __Torch__ and __CPU__ implementation: '
              f'\n\tSample size: {self.n_linear} '
              f'\n\tNumber of nodes: {self.d_linear}'
              f'\n\tNumber of edges: {self.s0_linear}'
              f'\n\tGraph type: {self.graph_type_linear}'
              f'\n\tSEM type: {self.sem_type_linear}\n')
        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'dagma',
                               compute_lib = 'torch', device = 'cpu', method_options = {'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for dagma (torch cpu): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testDagmaGauss3(self):

        X, W_true, B_true = self.generate_linear_data(self.n_linear,
                                                      self.d_linear,
                                                      self.s0_linear,
                                                      self.graph_type_linear,
                                                      self.sem_type_linear)
        # print the simulation information
        print(f'\n\nWorking with linear model with __Torch__ and __GPU__ implementation: '
              f'\n\tSample size: {self.n_linear} '
              f'\n\tNumber of nodes: {self.d_linear}'
              f'\n\tNumber of edges: {self.s0_linear}'
              f'\n\tGraph type: {self.graph_type_linear}'
              f'\n\tSEM type: {self.sem_type_linear}\n')
        
        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'dagma',
                               compute_lib = 'torch', device = 'cuda', method_options = {'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for dagma (torch gpu): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testDagmaLogistic1(self):

        # generate the data
        X, W_true, B_true = self.generate_linear_data(self.n_logistic,
                                                      self.d_logistic,
                                                      self.s0_logistic,
                                                      self.graph_type_logistic,
                                                      self.sem_type_logistic)
        # print the simulation information
        print(f'\n\nWorking with linear model with __Numpy__ implementation: '
              f'\n\tSample size: {self.n_logistic} '
              f'\n\tNumber of nodes: {self.d_logistic}'
              f'\n\tNumber of edges: {self.s0_logistic}'
              f'\n\tGraph type: {self.graph_type_logistic}'
              f'\n\tSEM type: {self.sem_type_logistic}\n')
        
        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'dagma', loss_fn = 'logistic', general_options={'lambda1':0.02},
                             method_options={'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for dagma numpy: ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testDagmaLogistic2(self):

        # generate the data
        X, W_true, B_true = self.generate_linear_data(self.n_logistic,
                                                      self.d_logistic,
                                                      self.s0_logistic,
                                                      self.graph_type_logistic,
                                                      self.sem_type_logistic)
        # print the simulation information
        print(f'\n\nWorking with linear model with __Torch__ and __CPU__ implementation: '
              f'\n\tSample size: {self.n_logistic} '
              f'\n\tNumber of nodes: {self.d_logistic}'
              f'\n\tNumber of edges: {self.s0_logistic}'
              f'\n\tGraph type: {self.graph_type_logistic}'
              f'\n\tSEM type: {self.sem_type_logistic}\n')
        
        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'dagma', loss_fn = 'logistic',
                               compute_lib = 'torch', device = 'cpu', general_options={'lambda1':0.02},
                               method_options = {'verbose': self.verbose})
        time_end = time.time()

        print('Time taken for dagma (torch cpu): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)
    
    def testDagmaLogistic3(self):
        # generate the data
        X, W_true, B_true = self.generate_linear_data(self.n_logistic,
                                                      self.d_logistic,
                                                      self.s0_logistic,
                                                      self.graph_type_logistic,
                                                      self.sem_type_logistic)
        # print the simulation information
        print(f'\n\nWorking with linear model with __Torch__ and __GPU__ implementation: '
              f'\n\tSample size: {self.n_logistic} '
              f'\n\tNumber of nodes: {self.d_logistic}'
              f'\n\tNumber of edges: {self.s0_logistic}'
              f'\n\tGraph type: {self.graph_type_logistic}'
              f'\n\tSEM type: {self.sem_type_logistic}\n')
        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'dagma', loss_fn = 'logistic',
                               compute_lib = 'torch', device = 'cuda', general_options={'lambda1':0.02},
                                 method_options={'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for dagma (torch gpu): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)
        
    # def testDagmaMLP1(self):
    #     X, B_true = self.generate_nonlinear_data(self.n_nonlinear,
    #                                              self.d_nonlinear,
    #                                              self.s0_nonlinear,
    #                                              self.graph_type_nonlinear,
    #                                              self.sem_type_nonlinear)

    #     print(f'\n\nWorking with nonlinear model with __torch__ and __CPU__ implementation: '
    #           f'\n\tSample size: {self.n_nonlinear} '
    #           f'\n\tNumber of nodes: {self.d_nonlinear}'
    #           f'\n\tNumber of edges: {self.s0_nonlinear}'
    #           f'\n\tGraph type: {self.graph_type_nonlinear}'
    #           f'\n\tSEM type: {self.sem_type_nonlinear}\n')
        
    #     time_start = time.time()
    #     W_est = dagrad(X = X, model = 'nonlinear', method = 'dagma', method_options={'verbose': self.verbose})
    #     time_end = time.time()
    #     print('Time taken for dagma torch cpu: ', time_end - time_start)
    #     acc = utils.count_accuracy(B_true, W_est != 0)
    #     print('\nAccuracy:', acc)
    #     print('-'*50)

    # def testDagmaMLP2(self):
    #     X, B_true = self.generate_nonlinear_data(self.n_nonlinear,
    #                                              self.d_nonlinear,
    #                                              self.s0_nonlinear,
    #                                              self.graph_type_nonlinear,
    #                                              self.sem_type_nonlinear)

    #     print(f'\n\nWorking with nonlinear model with __torch__ and __GPU__ implementation: '
    #           f'\n\tSample size: {self.n_nonlinear} '
    #           f'\n\tNumber of nodes: {self.d_nonlinear}'
    #           f'\n\tNumber of edges: {self.s0_nonlinear}'
    #           f'\n\tGraph type: {self.graph_type_nonlinear}'
    #           f'\n\tSEM type: {self.sem_type_nonlinear}\n')
        

    #     time_start = time.time()
    #     W_est = dagrad(X = X, model = 'nonlinear', method = 'dagma',
    #                            compute_lib = 'torch', device = 'cuda', method_options={'verbose': self.verbose})
    #     time_end = time.time()
    #     print('Time taken for dagma torch gpu: ', time_end - time_start)
    #     acc = utils.count_accuracy(B_true, W_est != 0)
    #     print('\nAccuracy: ', acc)
    #     print('-'*50)



class TestTOPO(unittest.TestCase):

    def setUp(self):
        # setup for linear model
        self.n_linear = 1000
        self.d_linear = 5
        self.s0_linear = 10
        self.graph_type_linear = 'ER'
        self.sem_type_linear = 'gauss'

        # setup for logisitic model
        self.n_logistic = 10000
        self.d_logistic = 5
        self.s0_logistic = 10
        self.graph_type_logistic = 'ER'
        self.sem_type_logistic = 'logistic'

        # setup for nonlinear model
        self.n_nonlinear = 1000
        self.d_nonlinear = 5
        self.s0_nonlinear = 10
        self.graph_type_nonlinear = 'ER'
        self.sem_type_nonlinear = 'mlp'

        # general setup
        self.verbose = False

    @staticmethod
    def generate_linear_data(n,d,s0,graph_type,sem_type,seed=1234):
        utils.set_random_seed(seed=seed)
        B_true = utils.simulate_dag(d, s0, graph_type)
        W_true = utils.simulate_parameter(B_true)
        X = utils.simulate_linear_sem(W_true, n, sem_type)
        return X, W_true, B_true
    @staticmethod
    def generate_nonlinear_data(n,d,s0,graph_type,sem_type,seed=1234):
        utils.set_random_seed(seed=seed)
        B_true = utils.simulate_dag(d, s0, graph_type)
        X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
        return X, B_true
    
    def testTopoGauss1(self):
        
        # generate the data
        X, W_true, B_true = self.generate_linear_data(self.n_linear,self.d_linear,self.s0_linear,self.graph_type_linear,self.sem_type_linear)

        # print the simulation information
        print(f'\n\nWorking with linear model with __sklearn__ solver: '
              f'\n\tSample size: {self.n_linear} '
              f'\n\tNumber of nodes: {self.d_linear}'
              f'\n\tNumber of edges: {self.s0_linear}'
              f'\n\tGraph type: {self.graph_type_linear}'
              f'\n\tSEM type: {self.sem_type_linear}\n')


        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'topo', method_options={'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for topo (sklearn): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testTopoGauss2(self):
        # generate the data
        X, W_true, B_true = self.generate_linear_data(self.n_linear,self.d_linear,self.s0_linear,self.graph_type_linear,self.sem_type_linear)

        # print the simulation information
        print(f'\n\nWorking with linear model with __LBFGS__ solver: '
              f'\n\tSample size: {self.n_linear} '
              f'\n\tNumber of nodes: {self.d_linear}'
              f'\n\tNumber of edges: {self.s0_linear}'
              f'\n\tGraph type: {self.graph_type_linear}'
              f'\n\tSEM type: {self.sem_type_linear}\n')
        
        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'topo',
                               optimizer = 'lbfgs', method_options={'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for topo (lbfgs): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testTopoLogistic1(self):
        # generate the data
        X, W_true, B_true = self.generate_linear_data(self.n_logistic,
                                                      self.d_logistic,
                                                      self.s0_logistic,
                                                      self.graph_type_logistic,
                                                      self.sem_type_logistic)
        # print the simulation information
        print(f'\n\nWorking with linear model with __sklearn__ solver: '
              f'\n\tSample size: {self.n_logistic} '
              f'\n\tNumber of nodes: {self.d_logistic}'
              f'\n\tNumber of edges: {self.s0_logistic}'
              f'\n\tGraph type: {self.graph_type_logistic}'
              f'\n\tSEM type: {self.sem_type_logistic}\n')
        
        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'topo', loss_fn = 'logistic', 
                             method_options={'verbose': self.verbose}, general_options={'lambda1':0.02})
        time_end = time.time()
        print('Time taken for topo (sklearn): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

    def testTopoLogistic2(self):
        
        X, W_true, B_true = self.generate_linear_data(self.n_logistic,
                                                      self.d_logistic,
                                                      self.s0_logistic,
                                                      self.graph_type_logistic,
                                                      self.sem_type_logistic)
        # print the simulation information
        print(f'\n\nWorking with linear model with __lbfgs__ implementation: '
              f'\n\tSample size: {self.n_logistic} '
              f'\n\tNumber of nodes: {self.d_logistic}'
              f'\n\tNumber of edges: {self.s0_logistic}'
              f'\n\tGraph type: {self.graph_type_logistic}'
              f'\n\tSEM type: {self.sem_type_logistic}\n')

        time_start = time.time()
        W_est = dagrad(X = X, model = 'linear', method = 'topo', loss_fn = 'logistic',
                               optimizer = 'lbfgs', method_options={'verbose': self.verbose}, general_options={'lambda1':0.02})
        time_end = time.time()
        print('Time taken for topo (lbfgs): ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)
        
    def testTopoMLP1(self):
        X, B_true = self.generate_nonlinear_data(self.n_nonlinear,
                                                 self.d_nonlinear,
                                                 self.s0_nonlinear,
                                                 self.graph_type_nonlinear,
                                                 self.sem_type_nonlinear)

        print(f'\n\nWorking with nonlinear model with __torch__ and __CPU__ implementation: '
              f'\n\tSample size: {self.n_nonlinear} '
              f'\n\tNumber of nodes: {self.d_nonlinear}'
              f'\n\tNumber of edges: {self.s0_nonlinear}'
              f'\n\tGraph type: {self.graph_type_nonlinear}'
              f'\n\tSEM type: {self.sem_type_nonlinear}\n')
        

        time_start = time.time()
        W_est = dagrad(X = X, model = 'nonlinear', method = 'topo', 
                             method_options={'verbose': self.verbose})
        time_end = time.time()
        print('Time taken for topo torch cpu: ', time_end - time_start)
        acc = utils.count_accuracy(B_true, W_est != 0)
        print('\nAccuracy: ', acc)
        print('-'*50)

if __name__ == '__main__':
    unittest.main()