import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm
from collections import namedtuple
import struct



'''
 /$$$$$$$  /$$             /$$    
| $$__  $$| $$            | $$    
| $$  \ $$| $$  /$$$$$$  /$$$$$$  
| $$$$$$$/| $$ /$$__  $$|_  $$_/  
| $$____/ | $$| $$  \ $$  | $$    
| $$      | $$| $$  | $$  | $$ /$$
| $$      | $$|  $$$$$$/  |  $$$$/
|__/      |__/ \______/    \___/  
                                                                  
'''
                                  
def plot_configs_from_file(filepath,i=-1,save=False):
    if hasattr(save,"__len__"):
        save = save
    elif save:
        save = filepath[:-4]
    else:
        save = False

    all_grids, *_ = open_configurations(filepath)

    plot_configs_from_grids(all_grids,i=i,save=save)


def plot_configs_from_grids(all_grids,i=-1,save=False):
    n_configs, L_2 = np.shape(all_grids)
    L = int(np.sqrt(L_2))

    n_X = 20
    if hasattr(i,"__len__"):
        n_to_print = len(i)
        n_Y = (n_to_print+n_X-1)//n_X
        
        fig = plt.figure(figsize=(n_X,n_Y))
        for j,ind in enumerate(i):
            grid_2D = np.reshape(all_grids[ind],(L,L))

            plt.subplot(n_Y, n_X, j+1)
            plt.matshow(grid_2D,0)
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

    elif i>0:
        plot_config(all_grids[i],save=save)
    else :
        n_Y = (n_configs+n_X-1)//n_X

        fig = plt.figure(figsize=(n_X,n_Y))
        for j in range(n_configs):
            grid_2D = np.reshape(all_grids[j],(L,L))
            plt.subplot(n_Y, n_X, j+1)
            plt.matshow(grid_2D,0)
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        if save:
            plt.savefig(save+'.png',bbox_inches='tight')
        plt.show()




def plot_config(grid_1D,save=False):
    L = int(np.sqrt(len(grid_1D)))
    grid_2D = np.reshape(grid_1D,(L,L))
    plt.matshow(grid_2D)
    plt.axis('off')
    if save:
        plt.savefig(save+'.png',bbox_inches='tight')
    plt.show()



'''
  /$$$$$$                                                     /$$              
 /$$__  $$                                                   | $$              
| $$  \__/  /$$$$$$  /$$$$$$$   /$$$$$$   /$$$$$$  /$$$$$$  /$$$$$$    /$$$$$$ 
| $$ /$$$$ /$$__  $$| $$__  $$ /$$__  $$ /$$__  $$|____  $$|_  $$_/   /$$__  $$
| $$|_  $$| $$$$$$$$| $$  \ $$| $$$$$$$$| $$  \__/ /$$$$$$$  | $$    | $$$$$$$$
| $$  \ $$| $$_____/| $$  | $$| $$_____/| $$      /$$__  $$  | $$ /$$| $$_____/
|  $$$$$$/|  $$$$$$$| $$  | $$|  $$$$$$$| $$     |  $$$$$$$  |  $$$$/|  $$$$$$$
 \______/  \_______/|__/  |__/ \_______/|__/      \_______/   \___/   \_______/

 '''


@njit(fastmath=True)
def step_metropolis(L,grid,boltz_coefs):
    ind = np.random.randint(0,L*L)
    count =  grid[ind%L + L*(((ind//L)+1)%L)] + grid[ind%L + L*(((ind//L)-1)%L)] + grid[((ind%L)-1)%L + L*((ind//L)%L)] + grid[((ind%L)+1)%L + L*((ind//L)%L)]
    if np.random.random() < boltz_coefs[count+5*grid[ind]]:
        grid[ind] = not (grid[ind])
    return grid

@njit(fastmath=True)
def sweep_metropolis(L,grid,boltz_coefs):
    for i in range(L*L):
        grid = step_metropolis(L,grid,boltz_coefs)
    return grid



def n_thermalization(n_therm,t,h):
    return n_therm

#   If gridInit use it, and thermalize
#   else init, if Nsample<0 reinit and thermalize at each generation
#   gridInit + Nsample<0 is a bit useless

@njit(fastmath=True)
def generate(L,T,H,N_configs,n_therm=100,n_samples=10, init_grid = 0):
    #kb = 1
    #J = 1.
    beta = 1./T
    
    bPure = np.exp(-2. * beta * np.arange(-4,5,2))
    bExte = np.exp(-2. * beta * H * np.arange(-1,2,2) )
    boltz = np.hstack((np.flip(bPure)*bExte[0],bPure*bExte[1]))



    if hasattr(init_grid,"__len__"):
        grid = init_grid
    elif init_grid == 1:
        grid = np.ones(L*L).astype(np.bool_)
    elif init_grid == -1:
        grid = np.zeros(L*L).astype(np.bool_)
    else :
        grid = np.random.randint(0,2,L*L).astype(np.bool_)

    all = np.zeros((N_configs,L*L)).astype(np.bool_)

# THERMALIZATION
    for i in range(n_therm):
        grid = sweep_metropolis(L,grid,boltz)
    all[0] = grid

# SAMPLING
    for i in range(N_configs-1):
        # Reinitialize grid to 0 and thermalize with n_therm sweeps
        if n_samples <= 0:
            if hasattr(init_grid,"__len__"):
                grid = init_grid
            elif init_grid==1:
                grid = np.ones(L*L).astype(np.bool_)
            elif init_grid == -1:
                grid = np.zeros(L*L).astype(np.bool_)
            else :
                grid = np.random.randint(0,2,L*L).astype(np.bool_)
                
            for j in range(n_therm):
                grid = sweep_metropolis(L,grid,boltz)
        # Decorrelate from previous grid with n_samples sweeps
        else:
            for j in range(n_samples):
                grid = sweep_metropolis(L,grid,boltz)
        all[i+1] = grid
    return all




@njit(parallel=True,fastmath=True)
def generate_parallel(L,T,h,n_configs,n_therm=100,init_grid = 0):
    #J = 1.
    #kb = 1
    beta = 1./T
    
    bPure = np.exp(-2. * beta * np.arange(-4,5,2))
    bExte = np.exp(-2. * beta * h * np.arange(-1,2,2) )
    boltz=np.hstack((np.flip(bPure)*bExte[0],bPure*bExte[1]))

    all = np.zeros((n_configs,L*L)).astype(np.bool_)

    random_reinit = False
    if hasattr(init_grid,"__len__"):
            grid0 = init_grid
    else:
        if init_grid == 1:
            grid0 = np.ones(L*L).astype(np.bool_)
        elif init_grid == -1:
            grid0 = np.zeros(L*L).astype(np.bool_)
        else :
            random_reinit = True
            grid0 = np.random.randint(0,2,L*L).astype(np.bool_)


    for i in prange(n_configs):
        if random_reinit:
            grid = np.random.randint(0,2,L*L).astype(np.bool_) 
        else :
            grid = grid0  
        for j in range(n_therm):
            grid = sweep_metropolis(L,grid,boltz)
        all[i] = grid

    return all



'''
 /$$$$$$$$ /$$ /$$          
| $$_____/|__/| $$          
| $$       /$$| $$  /$$$$$$ 
| $$$$$   | $$| $$ /$$__  $$
| $$__/   | $$| $$| $$$$$$$$
| $$      | $$| $$| $$_____/
| $$      | $$| $$|  $$$$$$$
|__/      |__/|__/ \_______/

'''

def save_configurations(filepath,all_grids, L,T,H,test=True):

    N_configs = len(all_grids)

    header_bytes = struct.pack(
    ">IIdd",   # int int double double
    L,N_configs,T,H)

    with open(filepath, "wb") as f:
        f.write(header_bytes)

        for config_int8 in np.packbits(all_grids,axis=1):
            f.write(config_int8.tobytes())
    if test:
        reopen, *_ = open_configurations(filepath)
        np.any(reopen!=all_grids)
    return True




def open_configurations(filepath):

    with open(filepath, "rb") as f:

        header_bytes = f.read(24)
        L, n_configs, T, H = struct.unpack(">IIdd",header_bytes)

        n_bytes_per_config = (L * L + 7) // 8

        all_grids = np.zeros((n_configs,L*L)).astype(np.bool_)
        i = 0
        while True:
            config_bytes = f.read(n_bytes_per_config)
            if not config_bytes:
                break

            config_bytes_np = np.frombuffer(config_bytes, dtype=np.uint8)
            all_grids[i] = np.unpackbits(config_bytes_np)[:L*L].astype(np.bool_)
            i+=1

    return all_grids,L,n_configs,T,H







'''
  /$$$$$$                      /$$                              
 /$$__  $$                    | $$                              
| $$  \ $$ /$$$$$$$   /$$$$$$ | $$ /$$   /$$  /$$$$$$$  /$$$$$$ 
| $$$$$$$$| $$__  $$ |____  $$| $$| $$  | $$ /$$_____/ /$$__  $$
| $$__  $$| $$  \ $$  /$$$$$$$| $$| $$  | $$|  $$$$$$ | $$$$$$$$
| $$  | $$| $$  | $$ /$$__  $$| $$| $$  | $$ \____  $$| $$_____/
| $$  | $$| $$  | $$|  $$$$$$$| $$|  $$$$$$$ /$$$$$$$/|  $$$$$$$
|__/  |__/|__/  |__/ \_______/|__/ \____  $$|_______/  \_______/
                                   /$$  | $$                    
                                  |  $$$$$$/                    
                                   \______/                     
'''

@njit(fastmath=True)
def magnetization(grid):
    return 2*np.sum(grid)/len(grid) - 1



@njit(fastmath=True)
def mean_magnetization(all_grids):
    n_configs = np.shape(all_grids)[0]
    Ms = np.zeros(n_configs)
    for i in range(n_configs):
        Ms[i] = magnetization(all_grids[i])
    return np.mean(Ms),np.std(Ms)


@njit(fastmath=True)
def corr(S,dx,dy):
    L = int(np.sqrt(len(S)))
    t = 0
    for x in range(L):
        for y in range(L):
            t+=S[x+L*y] * S[(x+dx)%L + L*((y+dy)%L)]
    return t/L**2


@njit(fastmath=True,parallel=True)
def correlation(grid,tol=1e-1):

    S = 2*grid-1
    L = int(np.sqrt(len(grid)))

    co_1D = np.empty(L*L).astype(np.float64)
    dist_1D = np.empty(L*L).astype(np.float64)

    for x in prange(L):
        for y in range(L):
            dist_1D[x+L*y] = np.sqrt(min(x,L-x)**2 + min(y,L-y)**2)
            co_1D[x+L*y] = corr(S,x,y)


    arg_dist = np.argsort(dist_1D)
    dist_1D = dist_1D[arg_dist]
    co_1D = co_1D[arg_dist]

    R,index,count = unique(dist_1D,tol=tol)
    correlation  = np.zeros(len(R))

    for i,indice in enumerate(index):
        correlation[i] = np.sum(co_1D[indice:indice+count[i]])/count[i]
    
    return R,correlation
    










'''
 /$$$$$$$$                  /$$          
|__  $$__/                 | $$          
   | $$  /$$$$$$   /$$$$$$ | $$  /$$$$$$$
   | $$ /$$__  $$ /$$__  $$| $$ /$$_____/
   | $$| $$  \ $$| $$  \ $$| $$|  $$$$$$ 
   | $$| $$  | $$| $$  | $$| $$ \____  $$
   | $$|  $$$$$$/|  $$$$$$/| $$ /$$$$$$$/
   |__/ \______/  \______/ |__/|_______/ 
                                         
                           
'''

@njit(fastmath=True)
def unique(tab,tol=5e-4):

    tempo = np.sort(tab.copy())
    out = np.empty(tempo.size, dtype=tempo.dtype)
    indices = np.empty(tempo.size, dtype=np.int32)
    counts = np.empty(tempo.size, dtype=np.int32)

    n = 0
    indice = 0

    prev = tempo[0]
    out[n] = prev
    indices[n] = indice

    count = 1
    n += 1
    

    for i in range(1, len(tempo)):

        if np.abs(tempo[i]-prev)>tol:
            prev = tempo[i]
            out[n] = prev
            counts[n-1] = count
            count=1
            indices[n] = i

            n += 1
        else :
            count+=1
    counts[n-1] = count
    return out[:n],indices[:n],counts[:n]
