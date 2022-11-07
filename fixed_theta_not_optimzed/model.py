import os
import sys
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import tensor
from torch import diag
from torch import Size
from torch import inverse
from torch import det
import torch.optim as optim

class Count_SVI:
    '''A class that implements stochastic variational inference for count data 
    using the poisson lognormal distribution with amortization'''
    def __init__(self, dtype=torch.float32):
        '''Initialize the model'''
        self.N = None # Number of data points
        self.Q = None # Number of features
        self.d = None # Dimension of the response vector
        self.K = None # Number of Latent factors
        self.X = None # Data matrix, X is NxQ matrix

        self.mu_true = None # True mean of the response vector, mu is dx1 vector
        self.L_true  = None # True L matrix, L is dxK matrix
        self.D_true  = None # True D matrix, D is dxd matrix
        self.B_true  = None # True B matrix, the matrix of regression weights, B is dxQ matrix
        self.sigma_true = None # True sigma matrix, sigma is a dxd matrix
        self.A_true = None # True A matrix, the matrix of cholesky decomposition, A is dxd matrix
        self.Y = None # Response vector, Y is Nxd matrix

        self.mu = None # Mean of the response vector, mu is dx1 vector
        self.L = None # L matrix, L is dxK matrix
        self.D = None # D matrix, D is dxd matrix
        self.B = None # B matrix, the matrix of regression weights, B is dxQ matrix
        
        self.sigma = None # sigma matrix, sigma is a dxd matrix
        self.A = None # A matrix, the matrix of cholesky decomposition, A is dxd matrix
        self.L_t = None # Transpose of L matrix, L_t is Kxd matrix
        self.sigma_inv = None # Inverse of sigma matrix, sigma_inv is a dxd matrix

        self.phi_1 = None # phi_1 is dx1 vector
        self.phi_2 = None # phi_2 is dx1 vector
        self.phi_3 = None # phi_3 is dx1 vector
        self.phi_4 = None # phi_4 is dx1 vector

        self.elbo = None # ELBO
        self.loss = None # Loss

        self.dtype = dtype

    def generate_data(self, N, Q, d, K, X, mu_true, L_true, D_true, B_true) -> None:
        '''Generate data according to the Poisson-Lognormal distribution'''
        self.N = N # Number of data points
        self.Q = Q # Number of features
        self.d = d # Dimension of the response vector
        self.K = K # Number of Latent factors
        self.X = X.to(self.dtype) # Data matrix, X is NxQ matrix
        self.mu_true = mu_true.to(self.dtype) # True mean of the response vector, mu is dx1 vector
        self.L_true  = L_true.to(self.dtype) # True L matrix, L is dxK matrix
        self.D_true  = D_true.to(self.dtype) # True D matrix, D is dxd matrix
        self.B_true  = B_true.to(self.dtype) # True B matrix, the matrix of regression weights, B is dxQ matrix
        assert self.X.shape       == Size([self.N, self.Q])
        assert self.mu_true.shape == Size([self.d, ])
        assert self.L_true.shape  == Size([self.d, self.K])
        assert self.D_true.shape  == Size([self.d, self.d])
        assert self.B_true.shape  == Size([self.d, self.Q]) 
        # sigma_true is semidefinite positive and D_true is a diagonal matrix
        # sigma = LL^T + D
        self.sigma_true = (L_true @ torch.t(L_true)) + D_true # sigma is a dxd matrix
        # Generate random matrices accoriding to eq 7
        self.A_true = torch.linalg.cholesky(self.sigma_true)
        Y = []
        for i in range(self.N):
            mu_i = self.mu_true + self.B_true @ self.X[i] # mu_i is a dx1 vector
            n = torch.distributions.multivariate_normal.MultivariateNormal(mu_i, self.sigma_true)
            # generate Poisson-Lognormal data
            z_i = n.sample()
            mu_yi = torch.exp(z_i)
            y = torch.poisson(torch.exp(mu_yi))
            Y.append(y)
        self.Y = torch.stack(Y) # Response vector, Y is Nxd matrix

    def optimizable_parameters(self) -> torch.Tensor:
        '''Return the optimizable parameters'''
        #yield self.mu
        #yield self.L
        #yield self.D
        #yield self.B
        yield self.phi_1
        yield self.phi_2
        yield self.phi_3
        yield self.phi_4

    def initialize_params(self) -> None:
        '''Initialize the parameters'''
        print("Initializing the parameters: \n")
        self.mu = torch.tensor([0.001])
        self.D = torch.tensor([[0.0200]])
        self.L = torch.tensor([[0.0800]])
        self.B = torch.zeros((d, Q))
        #self.mu = torch.randn((self.d, ), dtype=self.dtype, requires_grad=True)
        #self.L = torch.randn((self.d, self.K), dtype=self.dtype, requires_grad=True)
        #self.D = diag(torch.rand(self.d, dtype=self.dtype)).requires_grad_(True)
        #self.B = torch.randn((self.d, self.Q), dtype=self.dtype, requires_grad=True)
        self.L_t = torch.t(self.L)
        self.sigma = (self.L @ self.L_t + self.D)
        assert det(self.sigma) > 0, "The covariance matrix is not positive definite"
        self.sigma_inv = torch.inverse(self.sigma)
        self.A = torch.linalg.cholesky(self.sigma)
        self.phi_1 = torch.randn((self.d, ), dtype=self.dtype)/100
        self.phi_2 = torch.randn((self.d, ), dtype=self.dtype)/100
        self.phi_3 = torch.randn((self.d, ), dtype=self.dtype)/100
        self.phi_4 = torch.randn((self.d, ), dtype=self.dtype)/100
        self.phi_1.requires_grad = True
        self.phi_2.requires_grad = True
        self.phi_3.requires_grad = True
        self.phi_4.requires_grad = True
        #self.phi_1 = torch.randn((self.d, ), dtype=self.dtype, requires_grad=True) # phi_1 is a dx1 matrix
        #self.phi_2 = torch.randn((self.d, ), dtype=self.dtype, requires_grad=True) # phi_2 is a dx1 matrix
        #self.phi_3 = torch.randn((self.d, ), dtype=self.dtype, requires_grad=True) # phi_3 is a dx1 matrix
        #self.phi_4 = torch.randn((self.d, ), dtype=self.dtype, requires_grad=True) # phi_4 is a dx1 matrix
        self.elbo = torch.zeros(1, dtype=self.dtype)

    def set_train(self, batch_size=100, max_iter=1000, learning_rate=0.001) -> None:
        '''Initialize variables for monitoring of the training'''
        print("\nSetting up for training: \n")
        self.batch_size = batch_size
        self.n_iter = 0
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def set_history(self) -> None:
        '''Set the history of the model'''
        print("Setting the history: \n")
        # Set the history of the model parameters
        #self.mu_history = []
        #self.L_history = []
        #self.D_history = []
        #self.B_history = []

        # Set the history of the variational parameters
        self.phi_1_history = []
        self.phi_2_history = []
        self.phi_3_history = []
        self.phi_4_history = []

        # Set the history of the ELBO
        self.elbo_history = []

        # Set the history of the gradients
        #self.mu_grad_history = []
        #self.L_grad_history = []
        #self.D_grad_history = []
        #self.B_grad_history = []

        # Set the history of the gradients of the variational parameters
        self.phi_1_grad_history = []
        self.phi_2_grad_history = []
        self.phi_3_grad_history = []
        self.phi_4_grad_history = []

    def track_history(self) -> None:
        '''Track the history'''
        #self.mu_history.append(self.mu.data.numpy().copy())
        #self.L_history.append(self.L.data.numpy().copy())
        #self.D_history.append(self.D.data.numpy().copy())
        #self.B_history.append(self.B.data.numpy().copy())
        self.phi_1_history.append(self.phi_1.data.numpy().copy())
        self.phi_2_history.append(self.phi_2.data.numpy().copy())
        self.phi_3_history.append(self.phi_3.data.numpy().copy())
        self.phi_4_history.append(self.phi_4.data.numpy().copy())
        self.elbo_history.append(self.elbo.item())
        #self.mu_grad_history.append(self.mu.grad.numpy().copy())
        #self.L_grad_history.append(self.L.grad.numpy().copy())
        #self.D_grad_history.append(self.D.grad.numpy().copy())
        #self.B_grad_history.append(self.B.grad.numpy().copy())
        self.phi_1_grad_history.append(self.phi_1.grad.numpy().copy())
        self.phi_2_grad_history.append(self.phi_2.grad.numpy().copy())
        self.phi_3_grad_history.append(self.phi_3.grad.numpy().copy())
        self.phi_4_grad_history.append(self.phi_4.grad.numpy().copy())

    def view_params(self) -> None:
        #print("mu: ", self.mu, "\tmu_grad: ", self.mu.grad, "\n")
        #print("L: ", self.L, "\tL_grad: ", self.L.grad, "\n")
        #print("D: ", self.D, "\tD_grad: ", self.D.grad, "\n")
        #print("B: ", self.B, "\tB_grad: ", self.B.grad, "\n")
        print("phi1: ", self.phi_1, "\tphi1_grad: ", self.phi_1.grad, "\n")
        print("phi2: ", self.phi_2, "\tphi2_grad: ", self.phi_2.grad, "\n")
        print("phi3: ", self.phi_3, "\tphi3_grad: ", self.phi_3.grad, "\n")
        print("phi4: ", self.phi_4, "\tphi4_grad: ", self.phi_4.grad, "\n")


    def write_stats(self, file) -> None:
        '''Write the stats of the model to a file'''
        file.write(f"The true model parameters are: \n")
        file.write(f"N : {self.N}\n")
        file.write(f"Q : {self.Q}\n")
        file.write(f"d : {self.d}\n")
        file.write(f"K : {self.K}\n")
        file.write(f"X :\n{self.X}\n")
        file.write(f"mu :\n{self.mu_true}\n")
        file.write(f"L  :\n{self.L_true}\n")
        file.write(f"D  :\n{self.D_true}\n")
        file.write(f"sigma :\n{self.sigma_true}\n")
        file.write(f"A  :\n{self.A_true}\n")
        file.write(f"B  :\n{self.B_true}\n")
        file.write(f"Y :\n{self.Y}\n")
        file.write(f"The parameters after convergence are: \n")
        #file.write(f"mu :   \n{np.array(self.mu.data)}\nmu_grad: \n{np.array(self.mu.grad)}\n")
        #file.write(f"L  :   \n{np.array(self.L.data)}\nL_grad: \n{np.array(self.L.grad)}\n")
        #file.write(f"D  :   \n{np.array(self.D.data)}\nD_grad: \n{np.array(self.D.grad)}\n")
        #file.write(f"B  :   \n{np.array(self.B.data)}\nB_grad: \n{np.array(self.B.grad)}\n")
        file.write(f"phi_1 :{np.array(self.phi_1.data)}\nphi_1_grad: {np.array(self.phi_1.grad)}\n")
        file.write(f"phi_2 :{np.array(self.phi_2.data)}\nphi_2_grad: {np.array(self.phi_2.grad)}\n")
        file.write(f"phi_3 :{np.array(self.phi_3.data)}\nphi_3_grad: {np.array(self.phi_3.grad)}\n")
        file.write(f"phi_4 :{np.array(self.phi_4.data)}\nphi_4_grad: {np.array(self.phi_4.grad)}\n")
        file.write(f"sigma :{np.array(self.sigma.data)}\n")
        file.write(f"A  :  \n{np.array(self.A.data)}\n")
        file.write(f"ELBO: {self.elbo.item()}\n")
        file.write(f"The training parameters are: \n")
        file.write(f"batch_size : {self.batch_size}\n")
        file.write(f"max_iterations : {self.max_iter}\n")
        file.write(f"learning_rate : {self.learning_rate}\n")

    def write_history(self, file) -> None:
        '''Write the history of the model to a file'''
        file.write(f"History of the model: \n\n")
        file.write(f"Iterations : {self.n_iter}\n")
        file.write("\n")
        #file.write(f"mu history :    \n{repr(np.stack(self.mu_history))}\n\n")
        #file.write(f"L history :     \n{repr(np.stack(self.L_history))}\n\n")
        #file.write(f"D history :     \n{repr(np.stack(self.D_history))}\n\n")
        #file.write(f"B history :     \n{repr(np.stack(self.B_history))}\n\n")
        file.write(f"phi_1 history : \n{repr(np.stack(self.phi_1_history))}\n\n")
        file.write(f"phi_2 history : \n{repr(np.stack(self.phi_2_history))}\n\n")
        file.write(f"phi_3 history : \n{repr(np.stack(self.phi_3_history))}\n\n")
        file.write(f"phi_4 history : \n{repr(np.stack(self.phi_4_history))}\n\n")
        file.write(f"ELBO history :  \n{(self.elbo_history)}\n")
        #file.write(f"mu_grad history : \n{repr(np.stack(self.mu_grad_history))}\n\n")
        #file.write(f"L_grad history : \n{repr(np.stack(self.L_grad_history))}\n\n")
        #file.write(f"D_grad history : \n{repr(np.stack(self.D_grad_history))}\n\n")
        #file.write(f"B_grad history : \n{repr(np.stack(self.B_grad_history))}\n\n")
        file.write(f"phi_1_grad history : \n{repr(np.stack(self.phi_1_grad_history))}\n\n")
        file.write(f"phi_2_grad history : \n{repr(np.stack(self.phi_2_grad_history))}\n\n")
        file.write(f"phi_3_grad history : \n{repr(np.stack(self.phi_3_grad_history))}\n\n")
        file.write(f"phi_4_grad history : \n{repr(np.stack(self.phi_3_grad_history))}\n\n")
        file.write("\n")

    def sample_zn_reparametrized(self, n) -> torch.Tensor:
        '''Sample from z_n where z_n ~ N(mu_n, sigma_n)
        where mu_n is a dx1 vector and sigma_n is a dxd matrix
        sigma_n = (S_n + sigma^{-1})^{-1}
        m_n = sigma_n^{-1} * (S_n * m_n + sigma^{-1} * (mu + B*x_n))'''
        assert det((self.compute_Sn(n))) > 0, "The determinant of S_n is not positive"
        assert det((inverse(self.compute_Sn(n)) + (self.L @ self.L_t + self.D))), "The determinant of S_n is not positive"
        eps0 = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d)).sample().to(self.dtype)
        eps1 = torch.distributions.MultivariateNormal(torch.zeros(self.K), torch.eye(self.K)).sample().to(self.dtype)
        eps2 = torch.distributions.MultivariateNormal(torch.zeros(self.d), torch.eye(self.d)).sample().to(self.dtype)
        eps1_prime = torch.mv(self.L, eps1) + torch.mv(self.D, eps0) # Normally distributed with mean 0 and covariance sigma
        eps2_prime = torch.mv(diag(torch.reciprocal(torch.sqrt(diag(self.compute_Sn(n))))), eps2) # Normally distributed with mean 0 and S_n^{-1}
        
        # Applying algorithm 4 from reference 7: 
        # Fast Simulation of Hyperplane-Truncated Multivariate Normal Distributions
        # eps1_prime - (self.sigma) * np.linalg.inv(S_n_inv + self.sigma) (eps1_prime + eps2_prime)

        # To do : Speeding up the inversion using the Woodbury identity:
        zn_s = self.compute_mun(n) + eps1_prime - (self.L @ self.L_t + self.D) @ inverse(inverse(self.compute_Sn(n)) + (self.L @ self.L_t + self.D)) @ (eps1_prime + eps2_prime)
        # print("Reparame zn_s:", zn_s)
        return zn_s # zn_s is a dx1 vector
    
    def compute_mn(self, n) -> torch.Tensor:
        '''Create the m_n vector'''
        assert torch.all(torch.exp(self.phi_2) + self.Y[n, :] > 0) == tensor(True), "Logarithm is not defined for negative values"
        #print("m_n:",self.phi_1 * torch.log(torch.exp(self.phi_2) + self.Y[n, :]))
        return self.phi_1 * torch.log(torch.exp(self.phi_2) + self.Y[n, :])

    def compute_Sn(self, n) -> torch.Tensor:
        '''Create the S_n matrix'''
        assert torch.all(torch.exp(self.phi_4) + self.Y[n, :] > 0) == tensor(True), "Logarithm is not defined for negative values"
        # print("S_n:",diag(torch.exp((self.phi_3 * torch.log(torch.exp(self.phi_4) + self.Y[n, :])))))
        return diag(torch.exp((self.phi_3 * torch.log(torch.exp(self.phi_4) + self.Y[n, :]))))
    
    def compute_Sigman(self, n) -> torch.Tensor:
        '''Compute Sigma_n'''
        assert det((self.L @ self.L_t + self.D)) > 0, "The determinant of the matrix is not positive"
        assert det((self.compute_Sn(n) + inverse((self.L @ self.L_t + self.D)))) > 0, "The determinant of the matrix is not positive"
        #print("sigma_n:", inverse(self.compute_Sn(n) + inverse((self.L @ self.L_t + self.D))))
        return inverse(self.compute_Sn(n) + inverse((self.L @ self.L_t + self.D)))
    
    def compute_mun(self, n) -> torch.Tensor:
        assert det((self.L @ self.L_t + self.D)) > 0, "The determinant of the matrix is not positive"
        assert det(self.compute_Sigman(n)) > 0, "The determinant of the matrix is not positive"
        
        return inverse(self.compute_Sigman(n)) @ (self.compute_Sn(n) @ self.compute_mn(n) + inverse((self.L @ self.L_t + self.D)) @ (self.mu + self.B @ self.X[n]))

    def poisson_log_likelihood(self, y, z) -> torch.Tensor:
        '''Compute the log likelihood of the data given the poisson model'''
        # Compute the log likelihood of the data y given the poisson model with the mean parameter as exp^z
        # returns yz - exp(z) - log(y!), we skip the log(y!) term since it is constant 
        return torch.mul(y, z) - torch.exp(z)
    
    def normal_log_likelihood(self, y, mu, sigma) -> torch.Tensor:
        '''Compute the log likelihood of the data given the normal model'''
        # Compute the log likelihood of the data y given the normal model with the mean parameter as mu and covariance as sigma
        # returns -0.5 * ((y - mu) / sigma)^2 - 0.5 * log(2 * pi * sigma)
        assert sigma > 0, "The variance is not positive"
        return -0.5 * torch.square(torch.div((y - mu) , sigma)) - torch.log(sigma)

    def compute_elbo(self) -> torch.Tensor:
        '''Compute the evidence lower bound'''
        # Compute the evidence lower bound
        net_elbo = 0
        for n in range(self.N):
            for _ in itertools.repeat(None, self.batch_size):
                z_n = self.sample_zn_reparametrized(n)
                for i in range(self.d):
                    net_elbo = net_elbo + torch.div(self.poisson_log_likelihood(self.Y[n][i], z_n[i]), self.batch_size) - torch.div(self.normal_log_likelihood(z_n[i], self.compute_mun(n)[i], torch.reciprocal(self.compute_Sn(n)[i][i])), self.batch_size)
        return net_elbo

    def visualize_elbo(self) ->None:
        '''Visualize the convergence of the ELBO'''
        # Visualize the convergence of the ELBO
        plt.plot(np.array(self.elbo_history))
        plt.xticks(range(0, self.max_iter + 1, 1000))
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.savefig(f'elbo_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')

    def visualize_convergence(self) -> None:
        '''Visualize the convergence'''
        # Visualize the convergence of the ELBO
        plt.plot(np.array(self.elbo_history))
        plt.xticks(range(0, self.max_iter + 1, 1000))
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.savefig(f'elbo_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        plt.clf()

        plt.plot(np.array(self.phi_1_history))
        plt.xlabel('Iteration')
        plt.xticks(range(0, self.max_iter + 1, 1000))
        plt.ylabel('phi1')
        plt.savefig(f'phi1_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        plt.clf()

        plt.plot(np.array(self.phi_2_history))
        plt.xlabel('Iteration')
        plt.xticks(range(0, self.max_iter + 1, 1000))
        plt.ylabel('phi2')
        plt.savefig(f'phi2_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        plt.clf()

        plt.plot(np.array(self.phi_3_history))
        plt.xlabel('Iteration')
        plt.xticks(range(0, self.max_iter + 1, 1000))
        plt.ylabel('phi3')
        plt.savefig(f'phi3_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        plt.clf()

        plt.plot(np.array(self.phi_4_history))
        plt.xlabel('Iteration')
        plt.xticks(range(0, self.max_iter + 1, 1000))
        plt.ylabel('phi4')
        plt.savefig(f'phi4_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        plt.clf()

        plt.plot(np.array(self.phi_1_grad_history))
        plt.xlabel('Iteration')
        plt.xticks(range(0, self.max_iter + 1, 1000))
        plt.ylabel('phi1_grad')
        plt.savefig(f'phi1grad_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        plt.clf()

        plt.plot(np.array(self.phi_2_grad_history))
        plt.xlabel('Iteration')
        plt.ylabel('phi2_grad')
        plt.xticks(range(0, self.max_iter + 1, 1000))
        plt.savefig(f'phi2grad_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        plt.clf()

        plt.plot(np.array(self.phi_3_grad_history))
        plt.xlabel('Iteration')
        plt.xticks(range(0, self.max_iter + 1, 1000))
        plt.ylabel('phi3_grad')
        plt.savefig(f'phi3grad_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        plt.clf()

        plt.plot(np.array(self.phi_4_grad_history))
        plt.xlabel('Iteration')
        plt.xticks(range(0, self.max_iter + 1, 1000))
        plt.ylabel('phi4_grad')
        plt.savefig(f'phi4grad_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        plt.clf()

        #plt.plot(np.array(self.mu_history))
        #plt.xlabel('Iteration')
        #plt.xticks(range(0, self.max_iter + 1, 1000))
        #plt.ylabel('mu')
        #plt.savefig(f'mu_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        #plt.clf()

        #plt.plot(np.hstack(np.vstack(np.array(self.L_history))))
        #plt.xlabel('Iteration')
        #plt.xticks(range(0, self.max_iter + 1, 1000))
        #plt.ylabel('L')
        #plt.savefig(f'L_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        #plt.clf()

        #plt.plot(np.hstack(np.vstack(np.array(self.D_history))))
        #plt.xlabel('Iteration')
        #plt.xticks(range(0, self.max_iter + 1, 1000))
        #plt.ylabel('D')
        #plt.savefig(f'D_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        #plt.clf()

        #plt.plot(np.hstack(np.vstack(np.array(self.B_history))))
        #plt.xlabel('Iteration')
        #plt.xticks(range(0, self.max_iter + 1, 1000))
        #plt.ylabel('B')
        #plt.savefig(f'B_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        #plt.clf()

        #plt.plot(np.hstack(np.vstack(np.array(self.mu_grad_history))))
        #plt.xlabel('Iteration')
        #plt.xticks(range(0, self.max_iter + 1, 1000))
        #plt.ylabel('mu_grad')
        #plt.savefig(f'mu_grad_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        #plt.clf()

        #plt.plot(np.hstack(np.vstack(np.array(self.L_grad_history))))
        #plt.xlabel('Iteration')
        #plt.xticks(range(0, self.max_iter + 1, 1000))
        #plt.ylabel('L_grad')
        #plt.savefig(f'L_grad_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        #plt.clf()

        #plt.plot(np.hstack(np.vstack(np.array(self.D_grad_history))))
        #plt.xlabel('Iteration')
        #plt.xticks(range(0, self.max_iter + 1, 1000))
        #plt.ylabel('D_grad')
        #plt.savefig(f'D_grad_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        #plt.clf()

        #plt.plot(np.hstack(np.vstack(np.array(self.B_grad_history))))
        #plt.xlabel('Iteration')
        #plt.xticks(range(0, self.max_iter + 1, 1000))
        #plt.ylabel('B_grad')
        #plt.savefig(f'B_grad_{self.max_iter}_{self.learning_rate}_{self.batch_size}.png')
        #plt.clf()
    
    def train(self) -> None:
        '''Start training the model'''
        self.set_history()
        optimizer = optim.Adam(self.optimizable_parameters(), lr=self.learning_rate)
        for self.n_iter in tqdm(range(self.max_iter)):
            optimizer.zero_grad()
            self.elbo = self.compute_elbo()
            self.loss = -self.elbo
            self.loss.backward()
            self.track_history()
            self.view_params()
            optimizer.step()
        converged_file = open(f"converged_{self.max_iter}_{self.learning_rate}_{self.batch_size}.log", "w")
        self.write_stats(converged_file)
        converged_file.close()
        history_file = open(f"history_{self.max_iter}_{self.learning_rate}_{self.batch_size}.log", "w")
        self.write_history(history_file)
        history_file.close()
        self.visualize_elbo()
        self.visualize_convergence()
        print(f"\nTraining complete: \n")    

if __name__ == '__main__':
    # Reproducability
    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)

    # Set the model parameters
    N = 100
    Q = 1
    d = 1
    K = 1

    mu = torch.tensor([0.001])
    D = torch.tensor([[0.0200]])
    L = torch.tensor([[0.0800]])
    X = torch.tensor([
        [-4.47886662e-01], [  3.35587672e-02], [ -2.58525570e-01], [  8.47029158e-01], 
        [-1.91857240e-01], [  2.02302088e-01], [  6.35220529e-01], [ -8.60429031e-01], 
        [6.89084765e-01], [  7.26082384e-01], [  5.46519787e-01], [ -2.16796345e-01], 
        [-2.03131665e+00], [ -2.67117859e+00], [ -6.41955996e-01], [  5.67092211e-01], 
        [1.46860620e+00], [ -1.07426283e+00], [ -3.00284352e-01], [ -9.65569525e-01], 
        [1.69274881e+00], [  1.59007145e-02], [  1.78615546e+00], [ -5.99700511e-01], 
        [4.37390698e-01], [ -1.20501887e-01], [ -7.00298350e-01], [ -6.79311258e-02], 
        [2.14587705e-01], [  4.18088707e-01], [  7.88067860e-02], [  1.69563492e-01], 
        [-1.79487833e-01], [  1.83566371e-01], [  5.03653199e-01], [ -2.34302344e-01], 
        [-2.67028856e-01], [  7.48671587e-01], [ -8.20017548e-01], [  2.40321069e-01], 
        [6.43491834e-01], [ -2.37636380e-01], [ -2.27211895e-01], [  5.36227931e-01], 
        [1.49386271e-01], [  1.00057811e+00], [ -4.40557711e-01], [ -8.75806477e-01], 
        [5.38068107e-01], [ -8.26140714e-01], [  8.42443681e-01], [ -5.73898756e-01], 
        [1.63977953e+00], [ -4.67482847e-01], [  1.80991015e-01], [ -1.92435376e-02], 
        [-2.28134943e-01], [ -1.12608880e+00], [ -8.21905534e-01], [  1.52141132e+00], 
        [1.73467756e+00], [  3.50486242e-01], [  2.54622861e-01], [  1.85499293e+00], 
        [-4.68809154e-01], [ -2.94536755e-01], [ -4.46838975e-01], [  6.12600223e-02], 
        [8.14735677e-03], [  2.23626973e+00], [ -8.24867137e-01], [  4.56301184e-01], 
        [-2.24388383e+00], [ -1.62192915e-01], [ -1.08627600e+00], [  3.92308086e-01], 
        [1.69450501e+00], [ -1.03774625e+00], [ -1.63603765e+00], [ -8.19609955e-01], 
        [1.15116634e+00], [  1.23998143e+00], [  2.91247511e+00], [ -6.41585917e-01], 
        [-9.87252836e-01], [ -1.28676305e-01], [ -6.39897110e-02], [ -3.01800496e-01], 
        [-1.42750677e+00], [ -9.66868073e-01], [ -7.34904117e-01], [  3.94067321e-01], 
        [4.81321105e-01], [  5.69876733e-01], [ -2.79854091e-03], [ -9.61480902e-01], 
        [3.03040351e-02], [ -1.05020126e+00], [  9.14079161e-01], [ -9.13735806e-01]])
    B = torch.zeros((d, Q))
    model = Count_SVI()
    model.generate_data(N, Q, d, K, X, mu, L, D, B)
    model.initialize_params()
    model.set_train(batch_size=100, max_iter=500, learning_rate=0.0002)
    model.train()

