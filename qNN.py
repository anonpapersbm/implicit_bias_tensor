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
from optimizers import *
from infra import *

def output_A_NN(X):
    #inputs a m, n matrix with each row being a x
    #outputs a m, n,n tensor
    m,n = X.shape
    A = np.zeros((m,n,n))
    for i in range(m):
        temp = jnp.outer(X[i,:],X[i,:])
        temp = (temp+temp.T)/2
        A[i,:,:] = jnp.outer(X[i,:],X[i,:])

    return A


def loss_qNN(U,y,X):
    #U is n by d, y is m, X is m by n
    #output is a scalar
    y_pred = jnp.sum((U.T@X.T)**2,axis=0)
    
    return jnp.linalg.norm(y-y_pred)**2

def loss_qNN_A(U,y,A):
    #U is n by d, y is m, X is m by n
    #output is a scalar
    UU_top = U@U.T
    y_pred = jnp.einsum(A,(0,1,2),M_star,(1,2))
    
    return jnp.linalg.norm(y-y_pred)**2

def solve_qNN(problem, init_mag = 1e-1, init_point = None, lr=0.01, optimizer = None, plot_gradnorm = False, plot_loss = False, epochs=2000, loss_epsilon=1e-5, gradnorm_epsilon=1e-5):

    #d is the number of neurons we want to use
    d,y,X,loss_fnc = problem
    n = X.shape[1]
    
    if init_point is None:
        init_point = jax.random.normal(jax.random.PRNGKey(np.random.randint(0,2**31-1)), (n,d))
    #normalization
    init_point = init_mag * init_point/jnp.linalg.norm(init_point)

    if optimizer is not None:
        opt_state = optimizer.init(init_point)


    gradnorms = np.zeros(epochs)
    losses = np.zeros(epochs)

    l_g_fn = jax.value_and_grad(loss_fnc)
    U = init_point
    jit_l_g_fn = jax.jit(l_g_fn)
    # the main training loop
    for epoch in range(epochs):
        loss, grad = jit_l_g_fn(U,y,X)

        grad = np.nan_to_num(grad)

        gradnorms[epoch] = jnp.linalg.norm(grad)
        losses[epoch] = loss

        if loss < loss_epsilon:
            break
        if optimizer is None:
            U = U - lr*grad
        else:
            updates, opt_state = optimizer.update(grad, opt_state)
            U = optax.apply_updates(U, updates)
    
    if plot_gradnorm:
        plt.figure()
        plt.yscale("log")
        plt.plot(np.arange(epochs),gradnorms)

    if plot_loss:
        plt.figure()
        plt.yscale("log")
        plt.plot(np.arange(epochs),losses)
        

    return loss, U, (losses,gradnorms)
    

dist_to_gt_list = []
success_rate_list = []
lr = 1e-2
jit= True
init_mag = 0.1
iter_num=10000
r=1
level = 2

n_list = np.array([8,10,12])
m_list = np.array([20,30,40])

success_rate_map = np.zeros((len(n_list),len(m_list),2))

for (n,m),(idx1,idx2) in zip(product(n_list,m_list),product(np.arange(len(n_list)),np.arange(len(m_list)))):
    #sample x from gaussian distribution, and normalize X such that A has norm 1
    #m denotes number of observations, n denotes dimension of data


    # X = jax.random.normal(jax.random.PRNGKey(95),shape=(m,n))
    # X = 2*X/np.sqrt(jnp.linalg.norm(output_A_NN(X)))
    # A = output_A_NN(X)
    A = jax.random.normal(jax.random.PRNGKey(95),shape=(m,n,n))
    A = A/jnp.linalg.norm(A)

    #sample a true U from gaussian distribution, assumed to be 1
    U_star = jax.random.normal(jax.random.PRNGKey(1),shape=(n,r))
    U_star = U_star/jnp.sqrt(jnp.linalg.norm(U_star@U_star.T))
    M_star = U_star@U_star.T

    print(jnp.linalg.norm(U_star@U_star.T))
    print(jnp.linalg.norm(A))

    #generate y observations
    #y = jnp.sum((U_star.T@X.T)**2,axis=0)
    y = jnp.einsum(A,(0,1,2),M_star,(1,2))

    U = np.einsum(A,[0,1,2],y,[0])
    eigs, eigvec = jnp.linalg.eigh(np.kron(np.eye(r),U))
    v1 = eigvec[:,-1]
    w_0 = v1+ 0.01*np.random.randn(n*r)

    if r == 1:
        z_lifted = elevate_initialization(U_star,level)
    else:
        z_lifted = elevate_initialization(vec(U_star),level)

    trial_num = 10

    all_trajectories = []
    all_trajectories_lifted = []

    dist_to_gt = np.zeros((2,trial_num))

    ##############################
    #unlifted problem

    optimizer = optax.adam(1e-2)
    for trial in range(trial_num):
        _ , U_final, _ = solve_qNN((n,y,A,loss_qNN_A), init_mag=1e-2,optimizer=optimizer,epochs=iter_num)

        if r ==1:
            dist_to_gt[0,trial] =LA.norm(U_final@U_final.T-np.outer(U_star,U_star))
        else:
            dist_to_gt[0,trial] =jnp.linalg.norm(U_final@U_final.T-U_star@U_star.T)
    


    #############################
    #lifted problem 
    jit_level = level

    if jit == True:
        if r == 1:
            this_get_grad = jax.jit(get_grad)
            this_get_grad(z_lifted,z_lifted,A,level).block_until_ready()
            this_loss = jax.jit(loss_fnc)
            this_loss(z_lifted,z_lifted,A,level).block_until_ready()
            gd_loss = jax.jit(loss_fnc2)
            gd_loss(z_lifted,y,A,level).block_until_ready()
        else:
            this_loss = jax.jit(loss_func_highrank)
            this_loss(z_lifted,A,y,level).block_until_ready()
            this_get_grad = jax.jit(jax.grad(loss_func_highrank))
            this_get_grad(z_lifted,A,y,level).block_until_ready()
    else:
        this_get_grad = get_grad
        this_loss = loss_fnc

    

    #optimizer = customGD(lr*5,n,r,level,(A,y),loss=this_loss,g_thres=1e-3)
    #optimizer = vanillaGD(lr*10000)
    #optimizer = vanillaPGD(lr*10000, 1e-1, n , level)
    optimizer = optax.adam(lr*0.05)
    for trial in range(trial_num):
        if r==1:
            _, w_final_lifted, _ = solve((this_get_grad,this_loss),U_star, z_lifted, A, level, jax.random.PRNGKey(trial), init_mag*1e-3,optimizer=optimizer,epochs=iter_num,loss_epsilon=-np.inf)
        else:
            _, w_final_lifted, (loss_ter,grad_iter) = solve_highr((this_get_grad,this_loss), A,y,n,r, level, jax.random.PRNGKey(trial), init_mag*1e-5,optimizer=optimizer,epochs=20000,loss_epsilon=1e-13)
        
        dropped_sol = drop(w_final_lifted,epochs=2000)
        if r ==1:
            dist_to_gt[1,trial] = jnp.linalg.norm(np.outer(dropped_sol,dropped_sol)-np.outer(U_star,U_star))
        else:
            dist_to_gt[1,trial] = jnp.linalg.norm(unvec(dropped_sol,n,r)@unvec(dropped_sol,n,r).T-U_star@U_star.T)
    
    dist_to_gt_list.append(dist_to_gt)
    rate1 = len(dist_to_gt[0,dist_to_gt[0,:] <= 0.05])/len(dist_to_gt[0,:])
    rate2 = len(dist_to_gt[1,dist_to_gt[1,:] <= 0.05])/len(dist_to_gt[1,:])

    success_rate_map[idx1,idx2,0] = rate1
    success_rate_map[idx1,idx2,1] = rate2
