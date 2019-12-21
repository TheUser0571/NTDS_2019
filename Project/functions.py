import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def compute_laplacian(adjacency, normalize: bool):
    """ Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    D = np.diag(np.ravel(np.sum(adjacency, axis = 1).flatten()))
    L = D - adjacency
    if normalize:
        a = np.array(adjacency, dtype='float')
        a[np.sum(a, axis = 1)==0,0] = np.Inf
        Dn = np.diag(np.ravel(np.sum(a, axis = 1)) ** (-0.5))
        L = Dn.dot(L).dot(Dn)
    
    return L


def spectral_decomposition(laplacian: np.ndarray):
    """ Return:
        lamb (np.array): eigenvalues of the Laplacian
        U (np.ndarray): corresponding eigenvectors.
    """
    lamb, U = np.linalg.eigh(laplacian)
    return lamb, U


def plot_power_lines(lon, lat):
    fig = plt.gcf()
    ax = plt.gca()
    ax.plot(lon, lat, 'k', linewidth=0.3)


def plot_signal_on_graph(lon, lat, x=None, title='', vlim=[-0.03, 0.03]):
    fig = plt.gcf()
    ax = plt.gca()
    if x is not None:
        x = np.ravel(x)
    p = ax.scatter(lon, lat, c=x, marker='o',
                   s=7, cmap='RdBu_r', vmin=vlim[0], vmax=vlim[1])
    ax.dist = 7
    ax.set_axis_off()
    ax.set_title(title)
    if x is not None:
        fig.colorbar(p)
        
def GFT(U, signal):
    return U.T.dot(signal)
    
def iGFT(U, fourier_coefficients: np.ndarray):
    return U.dot(fourier_coefficients)