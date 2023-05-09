import jax
import jax.numpy as jnp
# from jax.config import config
# config.update("jax_enable_x64", True)
import optax
import numpy as np
from scipy import linalg as LA
from sklearn.cluster import KMeans
import time
from itertools import chain, repeat, product
import matplotlib.pyplot as plt



class vanillaGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def init(self, starting_point, lr = 0):
        if lr != 0:
            self.learning_rate = lr
        return self.learning_rate

    def update(self, gradients, opt_state):
        updates = -self.learning_rate * gradients
        return updates, opt_state

class vanillaPGD:
    def __init__(self, learning_rate, r, n , l, g_thres = 1e-5, t_thres = 200):
        self.learning_rate = learning_rate
        self.r = r
        self.n = n
        self.l = l
        self.g_thres = g_thres
        self.t_thres = t_thres

    def init(self, starting_point, lr=0):
        if lr != 0:
            self.learning_rate = lr
        return {'curr_iter':0, 't_noise':0}

    def update(self, gradients, opt_state):
        curr_iter = opt_state['curr_iter']+1
        t_noise = opt_state['t_noise']
        # if jnp.linalg.norm(gradients) < self.g_thres and curr_iter - t_noise > self.t_thres:
        #     t_noise = curr_iter
        #     randint = np.random.randint(0,2**31-1)
        #     noise = jax.random.normal(jax.random.PRNGKey(randint), shape=(self.n,))
        #     noise = elevate_initialization(noise, self.l)
        #     noise = self.r*noise / (jnp.sqrt(self.n*self.l)*jnp.linalg.norm(noise))
        #     updates = -self.learning_rate * (noise+gradients)
        # else:
        randint = np.random.randint(0,2**31-1)
        noise = jax.random.normal(jax.random.PRNGKey(randint), shape=(self.n,))
        noise = elevate_initialization(noise, self.l)
        noise = self.r*noise / (jnp.sqrt(self.n*self.l)*jnp.linalg.norm(noise))
        updates = -self.learning_rate * (noise+gradients)
        return updates, {'curr_iter':curr_iter, 't_noise':t_noise}

class customGD:
    def __init__(self, learning_rate, n, r, l, prob_params,loss,g_thres = 1e-5, buffer=100, beta=0.5, gamma=0.5, eta_0 = None):
        self.learning_rate = learning_rate
        self.n = n
        self.r = r
        self.l = l
        self.loss = loss
        self.A, self.b = prob_params #unlifted sensing matrices and measurements
        self.g_thres = g_thres #gradient threshold for saddle point detection
        self.escape_saddle = False
        self.buffer_limit = buffer
        self.buffer_step = 0
        self.beta = beta #for backtracking line search
        self.gamma = gamma #for backtracking line search
        if eta_0 is None:
            self.eta_0 = learning_rate #for backtracking line search
        else:
            self.eta_0 = eta_0


    def init(self, starting_point ,lr=0):
        if lr != 0:
            self.learning_rate = lr
        return {'curr_iter':0, 't_noise':0, 'curr_w':starting_point}

    def update(self, gradients, opt_state):
        curr_iter = opt_state['curr_iter']+1
        t_noise = opt_state['t_noise']
        curr_w = opt_state['curr_w']
        if jnp.linalg.norm(gradients) < self.g_thres and curr_iter > 100:
            if self.escape_saddle:
                t_noise = curr_iter
                dropped_x = jnp.nan_to_num(drop(curr_w,epochs=500)) #500 is good enough for n<10? But could be adjusted larger.
                dropped_x = unvec(dropped_x, self.n, self.r)
                xx_top = dropped_x@dropped_x.T
                grad_x = jnp.kron(jnp.eye(self.r),jnp.einsum(self.A,[0,1,2],xx_top,[1,2], self.A,[0,3,4]) - jnp.einsum(self.b, [0], self.A,[0,1,2]))
                eigs, eigvec = jnp.linalg.eigh(grad_x)
                direction = elevate_initialization(eigvec[:,0], self.l) #corresponding to the least eigenvalue (should be negatuve)
                #now we want to do backtracking line search
                this_eta = self.eta_0
                while self.loss(curr_w+this_eta*direction,self.A,self.b,self.l) > self.loss(curr_w,self.A,self.b,self.l) + self.beta*this_eta*jnp.inner(gradients.reshape(-1),direction.reshape(-1)):
                    this_eta = this_eta*self.gamma
                updates = this_eta * (direction)
                self.escape_saddle = False
            else:
                self.buffer_step += 1
                if self.buffer_step == self.buffer_limit:
                    self.escape_saddle = True
                    self.buffer_step = 0
                updates = -self.learning_rate * gradients
        else:
            self.escape_saddle = False
            updates = -self.learning_rate * gradients
        return updates, {'curr_iter':curr_iter, 't_noise':t_noise,'curr_w': curr_w+ updates}