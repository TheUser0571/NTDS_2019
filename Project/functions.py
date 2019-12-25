import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# -----------------------------------------------------------------------------------------------------------------


def compute_laplacian(adjacency, normalize: bool):
    """
    Return:
    L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    D = np.diag(np.ravel(np.sum(adjacency, axis=1).flatten()))
    L = D - adjacency
    if normalize:
        a = np.array(adjacency, dtype='float')
        a[np.sum(a, axis=1) == 0, 0] = np.Inf
        Dn = np.diag(np.ravel(np.sum(a, axis=1)) ** (-0.5))
        L = Dn.dot(L).dot(Dn)

    return L


# -----------------------------------------------------------------------------------------------------------------


def spectral_decomposition(laplacian: np.ndarray):
    """ Return:
        lamb (np.array): eigenvalues of the Laplacian
        U (np.ndarray): corresponding eigenvectors.
    """
    lamb, U = np.linalg.eigh(laplacian)
    return lamb, U


# -----------------------------------------------------------------------------------------------------------------


def plot_power_lines(lon, lat):
    """
    Plots the edges (=power lines) between the nodes.

    Plots a line for each edge.

    Parameters:
    lon (numpy.ndarray): longitude vector of shape (2, number of edges) containing the longitude coordinate of the starting nodes in the first line and of the ending nodes in the second line
    lat (numpy.ndarray): latitude vector of shape (2, number of edges)

    """
    fig = plt.gcf()
    ax = plt.gca()
    ax.plot(lon, lat, 'k', linewidth=0.3)


# -----------------------------------------------------------------------------------------------------------------


def plot_signal_on_graph(lon, lat, x=None, title='', vlim=[-0.03, 0.03]):
    """
   Plots signal on graph

   You have to create a figure before calling this function.

   Parameters:
   lon (pandas.core.series.Series) : longitude
   lat (pandas.core.series.Series) : latitude
   x :
   title :
   vlim :

   Returns:
   no returns

    """
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


# -----------------------------------------------------------------------------------------------------------------


def GFT(U, signal):
    """
    Fourier transform

    U : the discrete Fourier transform is a linear transformation. U is the matrix of that transformation
    signal : signal to be transformed to the frequency domain
    """
    return U.T.dot(signal)


# -----------------------------------------------------------------------------------------------------------------


def iGFT(U, fourier_coefficients: np.ndarray):
    """
    Inverse Fourier transform

    Parameters:
    U : inverse of the matrix for discrete Fourier transform
    fourier_coefficients : signal to be transformed from the frequency domain to the time domain

    """
    return U.dot(fourier_coefficients)


# -----------------------------------------------------------------------------------------------------------------

def magnitude_getter(signal):
    """
    Gets the components of the Fourier decomposition corresponding to the frequency of 1 day, 1 week and 1 year.
    """
    # n_sample = len(signal)
    # sampling_frequency = 1  # because we sample once an hour -> 24 times a day
    signal_hat = np.fft.fft(signal)
    sum=np.sum(np.abs(signal_hat))
    signal_hat = signal_hat/sum # normalization to be able to compare different nodes
    # f = np.arange(0, len(signal_hat) // 2, 1) * sampling_frequency / n_sample  # frequencies for the x axis
    # temp_day= np.abs(f - 1 / 24)
    # temp_week= np.abs(f- 1/(7*24))
    # temp_year= np.abs(f - 1 / (365 * 24))
    # index_day=np.argmin(temp_day)
    # index_week=np.argmin(temp_week)
    # index_year=np.argmin(temp_year)
    # print(index_year, index_week,index_day)
    # day_mag = np.abs(signal_hat[index_day-3:index_day+4])
    # week_mag = np.abs(signal_hat[index_week-3:index_week+4])
    # year_mag = np.abs(signal_hat[index_year-3:index_year+4])
    half_day_mag=np.abs(signal_hat[2192])
    day_mag = np.abs(signal_hat[1096])
    week_mag = np.abs(signal_hat[157])
    year_mag = np.abs(signal_hat[3])
    return half_day_mag, day_mag, week_mag, year_mag
