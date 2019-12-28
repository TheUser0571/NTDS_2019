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


def plot_signal_on_graph(lon, lat, x=None, title='', vlim=None):
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
    if vlim is None:
        if x is None:
            vlim=[0,0]
        else:
            vlim=[np.percentile(x, 10), np.percentile(x, 90)]
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


def plot_map(map_img):
    """
    Plots the map of europe as a background for the graph
    """
    fig = plt.gcf()
    ax = plt.gca()
    ax.imshow(map_img, extent=[-11.24360789326, 36.967974972, 35.3755584134, 60.2472037266], alpha=0.5)
    
    
def plot_forecast_actual(solar_fc, solar_ts, wind_fc, wind_ts, time_vector, start_time, end_time, node):
    """
    Plots the solar and wind forecast, actual and their difference 
    for 'node' from 'start_time' to 'end_time'
    """
    start_idx = int(np.where(time_vector==start_time)[0])
    end_idx = int(np.where(time_vector==end_time)[0])
    solar_diff = RPD(solar_fc, solar_ts)
    wind_diff = RPD(wind_fc, wind_ts)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(solar_fc[start_idx:end_idx, node], label='forecast')
    ax[0, 0].plot(solar_ts[start_idx:end_idx, node], label='actual')
    ax[0, 0].legend()
    ax[0, 0].set_xlabel('Time [h]')
    ax[0, 0].set_ylabel('MWh')
    ax[0, 0].set_title(f'Solar energy of node {node} from {start_time} to {end_time}')
    ax[0, 1].plot(wind_fc[start_idx:end_idx, node], label='forecast')
    ax[0, 1].plot(wind_ts[start_idx:end_idx, node], label='actual')
    ax[0, 1].legend()
    ax[0, 1].set_xlabel('Time [h]')
    ax[0, 1].set_ylabel('MWh')
    ax[0, 1].set_title(f'Wind energy of node {node} from {start_time} to {end_time}')
    ax[1, 0].plot(solar_diff[start_idx:end_idx, node], color='red', label='difference')
    ax[1, 0].legend()
    ax[1, 0].set_xlabel('Time [h]')
    ax[1, 0].set_ylabel('%')
    ax[1, 0].set_title(f'Solar difference between forecast and actual')
    ax[1, 1].plot(wind_diff[start_idx:end_idx, node], color='red', label='difference')
    ax[1, 1].legend()
    ax[1, 1].set_xlabel('Time [h]')
    ax[1, 1].set_ylabel('%')
    ax[1, 1].set_title(f'Wind difference between forecast and actual')
    plt.show()
    
def RPD(x, y):
    """
    Calculates the signed relative percetage difference between x and y
    if x is bigger than y, the RPD is positive
    if x is smaller than y, the RPD is negative
    """
    if x.shape != y.shape:
        print(f'Error: shape of x: {x.shape} is not equal to shape of y: {y.shape}')
        return None
    tmp = np.zeros(x.shape)
    tmp[(x + y) == 0] = 1
    x[tmp == 1] = 1
    out = 200 * (x - y) / (np.abs(x) + np.abs(y))
    out[tmp == 1] = 0
    return out

def plot_forecasting_on_graph(solar_fc_MWh, solar_ts_MWh, solar_diff, wind_fc_MWh, wind_ts_MWh, wind_diff, lon, lat, edge_list_lon, edge_list_lat, map_img):
    nb_years = int(solar_ts_MWh.shape[0]/(24*356))
    nb_nodes = solar_ts_MWh.shape[1]
    
    # average solar power
    plt.subplot(221)
    # the time series is too long to average in one go, so first calculate yearly average and then total
    tmp = np.zeros((nb_years, nb_nodes))
    for i in range(nb_years):
        tmp[i] = np.mean(solar_ts_MWh[i*24*365:(i+1)*24*365], axis=0)
    x = np.mean(tmp, axis=0)
    plot_map(map_img)
    plot_signal_on_graph(lon, lat, x, title='Average solar power [MWh]')
    plot_power_lines(edge_list_lon, edge_list_lat)
    
    # average wind power
    plt.subplot(222)
    tmp = np.zeros((nb_years, nb_nodes))
    for i in range(nb_years):
        tmp[i] = np.mean(wind_ts_MWh[i*24*365:(i+1)*24*365], axis=0)
    x = np.mean(tmp, axis=0)
    plot_map(map_img)
    plot_signal_on_graph(lon, lat, x, title='Average wind power [MWh]')
    plot_power_lines(edge_list_lon, edge_list_lat)
    
    # average solar forecasting error
    plt.subplot(223)
    tmp = np.zeros((nb_years, nb_nodes))
    for i in range(nb_years):
        tmp[i] = np.mean(np.abs(solar_diff[i*24*365:(i+1)*24*365]), axis=0)
    x = np.mean(tmp, axis=0)
    plot_map(map_img)
    plot_signal_on_graph(lon, lat, x, title='Average solar forecasting error [%]')
    plot_power_lines(edge_list_lon, edge_list_lat)
    
    # average wind forecasting error
    plt.subplot(224)
    tmp = np.zeros((nb_years, nb_nodes))
    for i in range(nb_years):
        tmp[i] = np.mean(np.abs(wind_diff[i*24*365:(i+1)*24*365]), axis=0)
    x = np.mean(tmp, axis=0)
    plot_map(map_img)
    plot_signal_on_graph(lon, lat, x, title='Average wind forecasting error [%]')
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.show()