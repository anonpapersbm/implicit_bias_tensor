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

def output_A_MS(mask):
    #outputs a n*n, n,n tensor
    n = mask.shape[0]
    A = np.zeros((n**2,n,n))
    for [i,j] in product(np.arange(n),repeat=2):
        temp = np.zeros((n,n))
        temp[i,j]=mask[i,j]
        temp = (temp+temp.T)/2
        A[i*n+j,:,:]=temp

    return A


dist_to_gt_list = []
lr = 0.02
jit= True
# init_mag = 0.1

n_list = np.array([5,8,10,12])
level_list = np.array([2])
init_mag_list = np.array([1e-3,1e-4,1e-5,1e-6,1e-7])

success_rate_map = np.zeros((len(n_list),len(init_mag_list),len(level_list),2))

for (n,init_mag,level),(idx1,idx2,idx3) in zip(product(n_list,init_mag_list,level_list),product(np.arange(len(n_list)),np.arange(len(init_mag_list)),np.arange(len(level_list)))):

    #for specific dimension
    z = np.zeros(n)
    for k in range(int(np.ceil(n/2))):
        z[2*k]=1

    M_star = np.outer(z,z)

    eps = 0.01
    mask = np.ones((n,n))*eps
    for i in range(n):
        mask[i,i]=1
        for k in (np.arange(int(np.floor(n/2)))+1):
            mask[i,2*k-1]=1
            mask[2*k-1,i]=1

    A = output_A_MS(mask)
    b = np.einsum(A,[0,1,2],M_star,[1,2])

    U = np.einsum(A,[0,1,2],b,[0])
    eigs, eigvec = jnp.linalg.eigh(U)
    v1 = eigvec[:,-1]
    w_0 = v1+ 0.01*np.random.randn(n)

    z_lifted = elevate_initialization(z,level)

    trial_num = 10

    all_trajectories = []
    all_trajectories_lifted = []

    dist_to_gt = np.zeros((2,trial_num))

    ##############################
    #unlifted problem
    jit_level= 0

    if jit == True:
        this_get_grad = jax.jit(get_grad)
        this_get_grad(z,z,A,0).block_until_ready()
        this_loss = jax.jit(loss_fnc)
        this_loss(z,z,A,0).block_until_ready()
    else:
        this_get_grad = get_grad
        this_loss = loss_fnc

    optimizer = vanillaGD(lr*2)
    for trial in range(trial_num):
        
        _, w_final,_= solve((this_get_grad,this_loss),z, z, A, 0, jax.random.PRNGKey(trial), init_mag, optimizer=optimizer,w_0=w_0)
        dist_to_gt[0,trial] = min(LA.norm(w_final-z),LA.norm(w_final+z))


    #############################
    #lifted problem 
    jit_level = level

    if jit == True:
        this_get_grad = jax.jit(get_grad)
        this_get_grad(z_lifted,z_lifted,A,level).block_until_ready()
        this_loss = jax.jit(loss_fnc)
        this_loss(z_lifted,z_lifted,A,level).block_until_ready()
        gd_loss = jax.jit(loss_fnc2)
        gd_loss(z_lifted,b,A,level).block_until_ready()
    else:
        this_get_grad = get_grad
        this_loss = loss_fnc


    optimizer = customGD(lr,n,level, (A,b),loss=gd_loss,eta_0=0.1)
    #optimizer = vanillaGD(lr*2)
    #optimizer = vanillaPGD(lr*2, 1e-3, n , level)
    for trial in range(trial_num):

        _, w_final_lifted,_= solve((this_get_grad,this_loss),z, z_lifted, A, level, jax.random.PRNGKey(trial), init_mag,optimizer=optimizer,w_0=w_0)

        """
        this is to skip the solving step by just using 0 as the solution
        """
        #w_final_lifted=elevate_initialization(jnp.zeros(n),level)

        dropped_sol = drop(w_final_lifted,epochs=2000)
        dist_to_gt[1,trial] = jnp.linalg.norm(np.outer(dropped_sol,dropped_sol)-M_star)
    
    dist_to_gt_list.append(dist_to_gt)
    rate1 = len(dist_to_gt[0,dist_to_gt[0,:] <= 0.05])/len(dist_to_gt[0,:])
    rate2 = len(dist_to_gt[1,dist_to_gt[1,:] <= 0.05])/len(dist_to_gt[1,:])
    # plt.figure()
    # plt.hist(dist_to_gt[0,:])
    # plt.figure()
    # plt.hist(dist_to_gt[1,:])
    success_rate_map[idx1,idx2,idx3,0] = rate1
    success_rate_map[idx1,idx2,idx3,1] = rate2

print(success_rate_map)