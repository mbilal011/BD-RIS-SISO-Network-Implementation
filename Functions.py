import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.special as sp


'''
#-------------------- description of variables ----------------------
Nt:         number of BS transmit antennas 
K:          number of single-antenna users 
nIRSrow:    number of rows of IRS elements 
nIRScol:    number of columns of IRS elements 
locU:       array of users' locations 
Lambda:     carrier wavelength   
kappa:      Rician factor 
xt:         x-coordiate of the center of tx ULA 
yt:         y-coordiate of the center of tx ULA 
zt:         z-coordiate of the center of tx ULA 
xs:         x-coordiate of the center of IRS UPA
ys:         y-coordiate of the center of IRS UPA 
zs:         z-coordiate of the center of IRS UPA
locT:       array of coordinates of the tx antennas 
locS:       array of coordinates of IRS elements 
dTU:        array of distance between tx antennas and user's antenna
dSU:        array of distance between IRS elements and user's antenna 
dTS:        array of distance between tx antennas and IRS elements 
alphaDir:   pathloss exponent for direct links 
alphaIRS:   pathloss exponent for IRS-related links 
betaTU:     pathloss for BS-users' links 
betaTS:     pathloss for BS-IRS links 
betaSU:     pathloss for IRS-user links 
hTU_LoS:    LoS conponent for BS-user links 
hTU_NLoS:   NLoS component for BS-user links 
hTU:        stack of tx-users' channel vectors 
hTS_LoS:    LoS conponent for BS-IRS links 
hTS_NLoS:   NLoS component for BS-IRS links 
hTS:        BS-IRS channel matrix 
hSU_LoS:    LoS conponent for IRS-user links 
hSU_NLoS:   NLoS component for IRS-user links 
hSU:        stack of IRS-users' channel vectors 
Gt:         transmit-antenna gain 
Gr:         receive-antenna gain
#--------------------------------------------------------------------            
'''

def generate_station_positions_2D(base_station_position):
    '''
    Input: Position as (x,y)
    Returns base station positions in the form of an array [(x,y)]
    '''
    xt, yt = base_station_position
    return np.array([[xt, yt]])


def generate_station_positions_3D(base_station_position: int):
    '''
    Input: Position as (x,y,z)
    Returns base station positions in the form of an array [(x,y,z)]
    '''
    xt, yt, zt = base_station_position
    return np.array([[xt, yt, zt]])

# def generate_user_positions_3D(num_pos: int, r_range: int, irs_position: tuple):
#     xs, ys, zs = irs_position
#     user_positions = []
#     for _ in range(num_pos):
#         theta = 2 * np.pi * random.random()  # Azimuth angle
#         phi = np.pi * random.random()       # Elevation angle
#         radius = r_range * np.sqrt(random.random())  # Use cbrt for 3D
#         x = xs + radius * np.cos(theta) * np.sin(phi)
#         y = ys + radius * np.sin(theta) * np.sin(phi)
#         z = 0
#         user_positions.append((x, y, z))
#     return np.array(user_positions)

# def generate_user_positions_3D(num_pos: int, r_range: int, irs_position: tuple):
#     xs, ys, zs = irs_position
#     user_positions = []
#     for _ in range(num_pos):
#         theta = 2 * np.pi * random.random()  # Azimuth angle
#         phi = np.pi * random.random()       # Elevation angle
#         radius = r_range + 30  # Ensure distance is exactly 30 units away
#         x = xs + radius * np.cos(theta) * np.sin(phi)
#         y = ys + radius * np.sin(theta) * np.sin(phi)
#         z = 0
#         user_positions.append((x, y, z))
#     return np.array(user_positions)

# Function to generate user positions
def generate_fixed_user_positions(K, radius, IRS_position):
    xs, ys, zs = IRS_position
    user_positions = []
    angle_increment = 360 / K 
    for i in range(K):
        angle = (i * angle_increment) + 20
        radian = np.deg2rad(angle) 
        x = xs + radius * np.cos(radian)
        y = ys + radius * np.sin(radian)
        z = 0
        user_positions.append((x, y, z))
    return np.array(user_positions)

# Function to generate circle and sectors data
def generate_circle_and_sectors_3D(IRS_position, radius, L):
    circle_radius = radius + 10
    x_IRS, y_IRS, z_IRS = IRS_position

    # Generate circle data (circle lies in the XY plane)
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = x_IRS + circle_radius * np.cos(theta)
    y_circle = y_IRS + circle_radius * np.sin(theta)
    z_circle = np.full_like(x_circle, 0)

    # Generate sectors data (lines originating from the IRS center)
    sector_lines = []
    if L > 1:
        angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
        for angle in angles:
            x_end = x_IRS + circle_radius * np.cos(angle)
            y_end = y_IRS + circle_radius * np.sin(angle)
            z_end = 0
            sector_lines.append(((x_IRS, y_IRS, 0), (x_end, y_end, z_end)))

    return x_circle, y_circle, z_circle, sector_lines

def generate_transmit_antenna_coordinates_2D(Nt: int, xt, yt, halfLambda, quarterLambda):

    '''
        Generates coordinates of all the transmit antennas, located half a wavelength parat on the same transmitter.
        Input : Nt = Number of antennas, xt = x coordinate, yt = y coordinate, half lambda = half wavelength
    '''
    locTcenter = np.array([xt, yt], dtype=float)
    locT = np.tile(locTcenter, (Nt, 1))
    if Nt % 2 == 0:
        locT[0, 1] = yt - 0.5 * (Nt - 2) * halfLambda - quarterLambda
    else:
        locT[0, 1] = yt - 0.5 * (Nt - 1) * halfLambda
    locT[:, 1] = [locT[0, 1] + nt * halfLambda for nt in range(Nt)]
    return locT

def generate_transmit_antenna_coordinates_3D(Nt, xt, yt, zt, halfLambda, quarterLambda):
    locTcenter = np.array([xt, yt, zt], dtype=float)
    locT = np.tile(locTcenter, (Nt, 1))
    if Nt % 2 == 0:
        locT[0, 1] = yt - 0.5 * (Nt - 2) * halfLambda - quarterLambda
    else:
        locT[0, 1] = yt - 0.5 * (Nt - 1) * halfLambda
    locT[:, 1] = [locT[0, 1] + nt * halfLambda for nt in range(Nt)]
    return locT

#  ---------------------------------------------------------New Functions Starting-----------------------------------------------------------------------------------

def generate_IRS_3D(IRS_position, L, edge_length):
    xs, ys, zs = IRS_position
    positions = []

    if L == 1:
        # Single face with no width in the y direction
        positions.append([xs - edge_length / 2, ys, zs])
        positions.append([xs + edge_length / 2, ys, zs])
    elif L == 2:
        positions.append([xs - edge_length / 2, ys + 0.5, zs])
        positions.append([xs + edge_length / 2, ys + 0.5, zs])
        positions.append([xs - edge_length / 2, ys - 0.5, zs])
        positions.append([xs + edge_length / 2, ys - 0.5, zs])
    else:
        radius = edge_length / (2 * np.sin(np.pi / L))  # Correct radius for a regular polygon
        for i in range(L):
            angle = i * 2 * np.pi / L
            vertex_xs = xs + radius * np.cos(angle)
            vertex_ys = ys + radius * np.sin(angle)
            positions.append([vertex_xs, vertex_ys, zs])
    
    return np.array(positions)

def generate_irs_coordinates_3D(start_vertex, end_vertex, nIRSrow, nIRScol, z_height):
    line_points = np.linspace(start_vertex, end_vertex, nIRScol)
    z_points = np.linspace(0, z_height, nIRSrow)
    
    coordinates = []
    for line_point in line_points:
        for z in z_points:
            coordinates.append([line_point[0], line_point[1], start_vertex[2] + z])
    return np.array(coordinates)

def generate_all_irs_coordinates(IRS_position, L, nIRSrow, nIRScol, edge_length, z_height):
    vertices = generate_IRS_3D(IRS_position, L, edge_length)
    all_coordinates = []
    face_midpoints = []

    if L == 1:
        start_vertex, end_vertex = vertices[:2]
        coordinates = generate_irs_coordinates_3D(start_vertex, end_vertex, nIRSrow, nIRScol, z_height)
        all_coordinates.append(coordinates)
        
        midpoint = (start_vertex + end_vertex) / 2
        face_midpoints.append(midpoint)
    elif L == 2:
        start_vertex_1, end_vertex_1, start_vertex_2, end_vertex_2 = vertices
        coordinates_1 = generate_irs_coordinates_3D(start_vertex_1, end_vertex_1, nIRSrow, nIRScol, z_height)
        coordinates_2 = generate_irs_coordinates_3D(start_vertex_2, end_vertex_2, nIRSrow, nIRScol, z_height)
        all_coordinates.append(coordinates_1)
        all_coordinates.append(coordinates_2)
        
        midpoint_1 = (start_vertex_1 + end_vertex_1) / 2
        midpoint_2 = (start_vertex_2 + end_vertex_2) / 2
        face_midpoints.append(midpoint_1)
        face_midpoints.append(midpoint_2)
    else:
        for i in range(L):
            start_vertex = vertices[i]
            end_vertex = vertices[(i + 1) % L]

            # Generate IRS coordinates along the line between the vertices with vertical displacement
            coordinates = generate_irs_coordinates_3D(start_vertex, end_vertex, nIRSrow, nIRScol, z_height)
            all_coordinates.append(coordinates)
            
            # Calculate midpoint of the face
            midpoint = (start_vertex + end_vertex) / 2
            face_midpoints.append(midpoint)
        
    all_coordinates = np.vstack(all_coordinates)
    face_midpoints = np.array(face_midpoints)

    return all_coordinates, vertices, face_midpoints

def calculate_distances_3D(locU, locT, face_midpoints, IRS_position, L):
    dTU = np.linalg.norm(locU - locT, axis=1)  # Distance from users to base station
    dTU = np.reshape(dTU, (-1, 1))

    # Calculate distances from each user to all IRS face midpoints
    distances_to_faces = np.linalg.norm(locU[:, np.newaxis] - face_midpoints[np.newaxis, :], axis=-1)

    # Find the minimum distance for each user to IRS faces
    closest_face_indices = np.argmin(distances_to_faces, axis=1)
    dSU = distances_to_faces[np.arange(len(locU)), closest_face_indices][:, np.newaxis]

    dTS = np.linalg.norm(locT - face_midpoints, axis=1)  # Distance from base station to IRS position
    dTS = np.reshape(dTS, (-1, 1))

    return dTU, dSU, dTS, closest_face_indices

def plot_edges(ax, vertices, nIRSrow, z_height):
    L = len(vertices)
    for i in range(L):
        start_vertex = vertices[i]
        end_vertex = vertices[(i + 1) % L]
        z_points = np.linspace(0, z_height, nIRSrow)

        # Draw vertical lines for the first and last z point
        ax.plot([start_vertex[0], start_vertex[0]],
                [start_vertex[1], start_vertex[1]],
                [start_vertex[2], start_vertex[2] + z_points[0]], 'r-')
        ax.plot([end_vertex[0], end_vertex[0]],
                [end_vertex[1], end_vertex[1]],
                [end_vertex[2], end_vertex[2] + z_points[0]], 'r-')
        
        ax.plot([start_vertex[0], start_vertex[0]],
                [start_vertex[1], start_vertex[1]],
                [start_vertex[2], start_vertex[2] + z_points[-1]], 'r-')
        ax.plot([end_vertex[0], end_vertex[0]],
                [end_vertex[1], end_vertex[1]],
                [end_vertex[2], end_vertex[2] + z_points[-1]], 'r-')

        # Draw horizontal lines at the first and last z point
        ax.plot([start_vertex[0], end_vertex[0]],
                [start_vertex[1], end_vertex[1]],
                [start_vertex[2] + z_points[0], end_vertex[2] + z_points[0]], 'r-')

        ax.plot([start_vertex[0], end_vertex[0]],
                [start_vertex[1], end_vertex[1]],
                [start_vertex[2] + z_points[-1], end_vertex[2] + z_points[-1]], 'r-')
        
def compute_path_gains(dTU, dSU, dTS, M, c0=1e-3, d0=1, aTU=3.5, aSU=2.0, aTS = 2.0):
    GTU = c0 * (dTU / d0) ** (-aTU)
    
    # Initialize GSU and GTS
    GSU = np.zeros((dSU.shape[0], M))
    GTS = np.zeros((dTS.shape[0], M))
    
    # Compute GSU for each user and each IRS element
    for k in range(dSU.shape[0]):
        for m in range(M):
            GSU[k, m] = c0 * (dSU[k, 0] / d0) ** (-aSU)
    
    # Compute GTS for each IRS face and each element
    for l in range(dTS.shape[0]):
        for m in range(M):
            GTS[l, m] = c0 * (dTS[l, 0] / d0) ** (-aTS)
    
    return GTU, GSU, GTS

def compute_channels_NOMA(GTU, GSU, GTS, K, L, M, K_factor):
    hRT = np.zeros(K, dtype=complex)
    hRI = np.zeros((K, M), dtype=complex)
    hIT = np.zeros((L, M), dtype=complex)

    for k in range(K):
        # Compute hRT (channel between BS and users)
        hRT_LoS = np.exp(1j * 2 * np.pi * np.random.rand())  # Deterministic LoS component
        hRT_NLoS = np.sqrt(1 / 2) * (np.random.randn() + 1j * np.random.randn())  # NLoS component
        hRT[k] = np.sqrt(GTU[k]) * (np.sqrt(K_factor / (1 + K_factor)) * hRT_LoS + np.sqrt(1 / (1 + K_factor)) * hRT_NLoS)

        # Compute hRI (channel between IRS and users)
        for m in range(M):
            hRI_LoS = np.exp(1j * 2 * np.pi * np.random.rand())  # Deterministic LoS component
            hRI_NLoS = np.sqrt(1 / 2) * (np.random.randn() + 1j * np.random.randn())  # NLoS component
            hRI[k, m] = np.sqrt(GSU[k, m]) * (np.sqrt(K_factor / (1 + K_factor)) * hRI_LoS + np.sqrt(1 / (1 + K_factor)) * hRI_NLoS)

    # Compute hIT (channel between IRS and BS)
    for l in range(L):
        for m in range(M):
            hIT_LoS = np.exp(1j * 2 * np.pi * np.random.rand())  # Deterministic LoS component
            hIT_NLoS = np.sqrt(1 / 2) * (np.random.randn() + 1j * np.random.randn())  # NLoS component
            hIT[l, m] = np.sqrt(GTS[l, m]) * (np.sqrt(K_factor / (1 + K_factor)) * hIT_LoS + np.sqrt(1 / (1 + K_factor)) * hIT_NLoS)
    
    hRT = hRT.reshape(-1, 1)  # Shape: (K, 1)
    hRI = hRI.transpose()    # Shape: (M, K)
    
    return hRT, hRI, hIT
        
def generate_rician_channel_amplitude(K_factor, shape):
    # Generate LOS component amplitude
    los_component_amplitude = np.sqrt(K_factor / (K_factor + 1))
    
    # Generate NLOS component amplitude
    sigma = 1 / np.sqrt(2 * (K_factor + 1))
    nlos_component_amplitude = sigma * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
    
    # Combine LOS and NLOS components to get the amplitude
    return los_component_amplitude + nlos_component_amplitude

def generate_phase_shift(shape):
    return np.exp(-1j * 2 * np.pi * np.random.rand(*shape))

def compute_channels(K, L, M, K_factor):
    hRT = np.zeros(K, dtype=complex)
    hRI = np.zeros((K, M), dtype=complex)
    hIT = np.zeros((L, M), dtype=complex)

    for k in range(K):
        # Compute hRT (channel between BS and users)
        amplitude = generate_rician_channel_amplitude(K_factor, ())
        phase_shift = generate_phase_shift(())
        hRT[k] = amplitude * phase_shift

        # Compute hRI (channel between IRS and users)
        for m in range(M):
            amplitude = generate_rician_channel_amplitude(K_factor, ())
            phase_shift = generate_phase_shift(())
            hRI[k, m] = amplitude * phase_shift

    # Compute hIT (channel between IRS and BS)
    for l in range(L):
        for m in range(M):
            amplitude = generate_rician_channel_amplitude(K_factor, ())
            phase_shift = generate_phase_shift(())
            hIT[l, m] = amplitude * phase_shift
    
    hRT = hRT.reshape(-1, 1)
    hRI = hRI.transpose()
    
    return hRT, hRI, hIT

def generate_large_scale_fading(dTS, dSU, Gt, Gr, wavelength, L, eta_RIS, eta_kl, K, closest_face_indices):
    numerator = (wavelength**4 * Gt * Gr)
    alpha_kl = np.zeros((K, L))

    if L == 1:
        dTS_l = dTS[0]  # Distance from BS to the single face
        for k in range(K):
            dSU_k = dSU[k]  # Distance from face to user k
            denominator = (4**3 * np.pi**4 * dTS_l**eta_RIS * dSU_k**eta_kl)
            alpha_kl[k, 0] = numerator / denominator
    else:
        for k in range(K):
            l = closest_face_indices[k]  # Closest face for user k
            dTS_l = dTS[l]  # Distance from BS to face l
            dSU_k = dSU[k]  # Distance from face l to user k
            denominator = (4**3 * np.pi**4 * dTS_l**eta_RIS * dSU_k**eta_kl * (1 - np.cos(np.pi / L))**2)
            alpha_kl[k, l] = numerator / denominator

    return alpha_kl

def generate_large_scale_fading_new(dTS, dSU, L, K, closest_face_indices):
    numerator = 1
    alpha_kl = np.zeros((K, L))

    if L == 1:
        dTS_l = dTS[0]  # Distance from BS to the single face
        for k in range(K):
            dSU_k = dSU[k]  # Distance from face to user k
            denominator = 1
            alpha_kl[k, 0] = numerator / denominator
    else:
        for k in range(K):
            l = closest_face_indices[k]  # Closest face for user k
            dTS_l = dTS[l]  # Distance from BS to face l
            dSU_k = dSU[k]  # Distance from face l to user k
            denominator = (1 - np.cos(np.pi / L))**2
            alpha_kl[k, l] = numerator / denominator

    return alpha_kl

# Define functions for SEP  
def q_function(x):
    return 0.5 * sp.erfc(x / np.sqrt(2))

def compute_sep_mpsk(snr_linear, B):
    # Ensure non-negative SNR values
    snr_linear = np.maximum(snr_linear, 1e-10)
    sep = 2 * q_function(np.sqrt(2 * snr_linear * np.sin(np.pi / B)**2))
    return sep

#  ---------------------------------------------------------New Functions Ending-----------------------------------------------------------------------------------

# Function to compute outage probability at each iteration
def compute_outage_probability(num_users, rate, rate_threshold):
    outage = 0
    for j in range(num_users):
      outage = np.sum(rate[j] < rate_threshold)
      return outage / num_users


# Function to compute average outage probability
def compute_average_outage_probability(outage_probabilities):
    num_simulations = len(outage_probabilities)
    outage_prob_sum = np.sum(outage_probabilities)
    return outage_prob_sum / num_simulations

# Function to compute outage probability at each iteration
def compute_energy_efficiency(rate, power):
    return rate / power


# Function to compute average outage probability
def compute_average_energy_efficiency(ee):
    num_simulations = len(ee)
    ee_sum = np.sum(ee)
    return ee_sum / num_simulations


def compute_rate(SNR):
    SNR_watts = (10**(SNR/10))
    return np.log2(1 + SNR_watts)

def compute_rate_NOMA(SNR):
    return np.log2(1 + SNR)

def calc_link_budget(rayleigh_channel, distance, path_loss_exponent, transmit_power):
        link_inter = (((np.abs(rayleigh_channel)) / np.sqrt((distance) ** path_loss_exponent)) ** 2) * (transmit_power)
        link_budget = 10 * np.log10(link_inter) + 30 #need to add actual noise power
        return link_budget

def compute_noise(noise_floor, bandwidth):
    k = 1.38 * 10 ** (-23)
    T = 290
    NOISE_POWER = k*T*bandwidth*noise_floor
    return NOISE_POWER

def compute_path_loss(distances, path_loss_exponent):
    return 1 / np.sqrt(distances ** path_loss_exponent)

def generate_rayleigh_fading_channel(K, std_mean, std_dev):
    X = np.random.normal(std_mean, std_dev, K) 
    Y = np.random.normal(std_mean, std_dev, K) 
    rayleigh_channel = (X + 1j*Y)
    return rayleigh_channel

def generate_nakagami_samples(m, omega, size):
    magnitude_samples = np.sqrt(omega) * np.sqrt(np.random.gamma(m, 1, size)) / np.sqrt(np.random.gamma(m - 0.5, 1, size))
    phase_samples = np.random.uniform(0, 2 * np.pi, size=size)
    complex_samples = magnitude_samples * np.exp(1j * phase_samples)
    return complex_samples

def compute_SNR(link_budget, noise_floor):
    SNR = link_budget - noise_floor
    return SNR

def compute_SNR_NOMA(link_budget, noise_floor):
    SNR = link_budget - noise_floor
    return dBm2pow(SNR)

def wrapTo2Pi(theta):
    return np.mod(theta,2*np.pi)

def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

#function for converting watts to dBm
def pow2dBm(watt):
    dBm = 10* np.log10(watt) + 30
    return dBm
    
#function for converting dBm to watts
def dBm2pow(dBm):
    watt = (10**(dBm/10))/1000
    return watt

def db2pow(dB):
    watt = (10**(dB/10))
    return watt

def pow2db(watt):
    db = 10 * np.log10(watt)
    return db

def generate_quantized_theta_set(B):
    K = 2**B
    delta_theta = 2 * np.pi / K
    quantized_theta_set = np.arange(0, K) * delta_theta - np.pi
    return quantized_theta_set

def compute_results_array_continuous(K, Ns, Nt, h_dk, h_rk, h_rk_transpose, G, d, d_max):
    # Initialize empty lists to store theta_n values and results
    theta_n_values_complex = []

    for i in range(K):
        theta_n_i = []
        for j in range(Ns):
            theta_n = np.angle(h_dk[0][i]) - np.angle(h_rk[j][i]) - np.angle(G[j][0])
            theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi
            theta_n_i.append(theta_n)
        theta_n_values_complex.append(1 * np.exp(1j * np.array(theta_n_i)))
        
    theta_n_values_complex = np.array(theta_n_values_complex)

    # Initialize an empty list to store diagonal matrices
    diagonal_matrices = []

    for row in theta_n_values_complex:
        diagonal_matrix = np.diag(row[:Ns])
        diagonal_matrices.append(diagonal_matrix)

    # Convert diagonal_matrices to a NumPy array
    diagonal_matrices = np.array(diagonal_matrices)

    # Initialize an empty list to store the results for each column
    results_list = []

    for row_index in range(diagonal_matrices.shape[0]):
        single_row_diag = diagonal_matrices[row_index, :, :]
        single_row = h_rk_transpose[row_index,:]
    
        result_inter = np.dot(single_row, single_row_diag)

        result = np.dot(result_inter, G)
        results_list.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results_list)
    results_array = results_array.reshape(Nt, K)

    return results_array

def results_array_discrete(K, Ns, Nt, h_dk, h_rk, h_rk_transpose, G, B):
    # Create a set of quantized theta values
    quantized_theta_set = ((2 * np.pi * np.arange(0, 2**B, 1) / (2**B)) - np.pi)
    quantized_theta_n_values_complex = []

    for i in range(K):
        quantized_theta_n_i = []

        for j in range(Ns):
            theta_n = np.angle(h_dk[0][i]) - np.angle(h_rk[j][i]) - np.angle(G[j][0])
            theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi
            nearest_quantized_theta = quantized_theta_set[np.argmin(np.abs(theta_n - quantized_theta_set))]
            quantized_theta_n_i.append(nearest_quantized_theta)

        quantized_theta_n_values_complex.append(1 * np.exp(1j * np.array(quantized_theta_n_i)))

    theta_n_values_complex = np.array(quantized_theta_n_values_complex)

    # Initialize an empty list to store diagonal matrices
    diagonal_matrices = []

    # Transform each row into a diagonal matrix
    for row in theta_n_values_complex:
        diagonal_matrix = np.diag(row[:Ns])
        diagonal_matrices.append(diagonal_matrix)

    # Convert diagonal_matrices to a NumPy array
    diagonal_matrices = np.array(diagonal_matrices)

    # Initialize an empty list to store the results for each column
    results_list = []

    # Loop over each row/user in the diagonal_matrices
    for row_index in range(diagonal_matrices.shape[0]):
        single_row_diag = diagonal_matrices[row_index, :, :]
        single_row = h_rk_transpose[row_index,:]
        result_inter = np.dot(single_row, single_row_diag)
        result = np.dot(result_inter, G)
        results_list.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results_list)
    results_array = results_array.reshape(Nt, K)

    return results_array

def results_array_practical_discrete(K, Ns, Nt, h_dk, h_rk, h_rk_transpose, G, B, beta_min, k, phi):
    # Create a set of quantized theta values
    quantized_theta_set = ((2 * np.pi * np.arange(0, 2**B, 1) / (2**B)) - np.pi)

    # Initialize an empty list to store quantized theta_n values for each i
    quantized_theta_n_values_complex = []

    for i in range(K):
        beta_n = []
        quantized_theta_n_i = []

        for j in range(Ns):
            theta_n = - np.angle(h_rk[j][i]) - np.angle(G[j][0])

            # Adjust theta_n to lie within the range (-π, π)
            theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi

            # Find the nearest quantized theta value
            nearest_quantized_theta_new = quantized_theta_set[np.argmin(np.abs(theta_n - quantized_theta_set))]
            quantized_theta_n_i.append(nearest_quantized_theta_new)

            beta_theta_n = ((1 - beta_min) * ((np.sin(nearest_quantized_theta_new - phi) + 1) / 2) ** k + beta_min)
            beta_n.append(beta_theta_n)

        quantized_theta_n_values_complex.append(np.array(beta_n) * np.exp(1j * np.array(quantized_theta_n_i)))

    theta_n_values_complex = np.array(quantized_theta_n_values_complex)

    # Initialize an empty list to store diagonal matrices
    diagonal_matrices = []

    # Transform each row into a diagonal matrix
    for row in theta_n_values_complex:
        diagonal_matrix = np.diag(row[:Ns])
        diagonal_matrices.append(diagonal_matrix)

    # Convert diagonal_matrices to a NumPy array
    diagonal_matrices = np.array(diagonal_matrices)

    # Initialize an empty list to store the results for each column
    results_list = []

    # Loop over each row/user in the diagonal_matrices
    for row_index in range(diagonal_matrices.shape[0]):
        # Get the corresponding diagonal matrix for the current row/user
        single_row_diag = diagonal_matrices[row_index, :, :]

        # Extract the single column from f_m_transpose using indexing and transpose
        single_row = h_rk_transpose[row_index,:]

        # Perform the dot product between f_m_transpose (5, 10) and the current diagonal matrix (10, 10)
        result_inter = np.dot(single_row, single_row_diag)

        # Perform the final matrix multiplication of the result_inter (5, 10) and g (10, 1)
        result = np.dot(result_inter, G)
        results_list.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results_list)
    results_array = results_array.reshape(Nt, K)

    return results_array

def results_array_sharing_ideal(K, Ns, Nt, h_dk, h_rk, h_rk_transpose, G):
    # Initialize an empty list to store theta_n values for each i
    theta_n_values_complex = []
    inc = int(Ns / K)

    for i in range(K):
        theta_n_i = []

        for j in range(i * inc, (i + 1) * inc):
            theta_n = np.angle(h_dk[0][i]) - np.angle(h_rk[j][i]) - np.angle(G[j][0])

            # Adjust theta_n to lie within the range (-π, π)
            theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi
            theta_n_i.append(theta_n)

        theta_n_values_complex.append(1 * np.exp(1j * np.array(theta_n_i)))

    theta_n_values_complex = np.array(theta_n_values_complex)

    # Initialize an empty list to store diagonal matrices
    diagonal_matrices = []

    # Transform each row into a diagonal matrix
    for row in theta_n_values_complex:
        diagonal_matrix = np.diag(row[:Ns])
        diagonal_matrices.append(diagonal_matrix)

    # Convert diagonal_matrices to a NumPy array
    diagonal_matrices = np.array(diagonal_matrices)
    # print(np.shape(diagonal_matrices))

    results_list = []

    for row_index in range(diagonal_matrices.shape[0]):
        # Get the corresponding diagonal matrix for the current row/user is 1x1
        single_row_diag = diagonal_matrices[row_index]

        # Calculate the starting and ending indices for slicing based on row_index
        start_index = row_index * inc
        end_index = start_index + inc

        # Extract the single column from h_rk_transpose using slicing and transpose
        single_row = h_rk_transpose[row_index, start_index:end_index]

        # Reshape the single_row to (1, inc)
        single_row = single_row.reshape(1, inc)

        # Perform the dot product between f_m_transpose (1, inc) and the current diagonal matrix (inc, inc)
        result_inter = np.dot(single_row, single_row_diag)

        # Perform the final matrix multiplication of result_inter (1, inc) and a subset of G (inc, 1)
        subset_G = G[start_index:end_index]
        result = np.dot(result_inter, subset_G)
        results_list.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results_list)
    results_array = results_array.reshape(Nt, K)

    return results_array

def results_array_sharing_practical(K, Ns, Nt, h_dk, h_rk, h_rk_transpose, G, B, beta_min, phi, k):
    # Create a set of quantized theta values
    quantized_theta_set = ((2 * np.pi * np.arange(0, 2**B, 1) / (2**B)) - np.pi)

    # Initialize an empty list to store theta_n values for each i
    theta_n_values_complex = []
    inc = int(Ns / K)

    for i in range(K):
        theta_n_i = []
        beta_n = []

        for j in range(inc * i, inc * (i + 1)):
            theta_n = np.angle(h_dk[0][i])- np.angle(h_rk[j][i]) - np.angle(G[j][0])

            # Adjust theta_n to lie within the range (-π, π)
            theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi

            # Find the nearest quantized theta value
            nearest_quantized_theta_new = quantized_theta_set[np.argmin(np.abs(theta_n - quantized_theta_set))]
            theta_n_i.append(nearest_quantized_theta_new)

            beta_theta_n = ((1 - beta_min) * ((np.sin(nearest_quantized_theta_new - phi) + 1) / 2) ** k + beta_min)
            beta_n.append(beta_theta_n)

        theta_n_values_complex.append(np.array(beta_n) * np.exp(1j * np.array(theta_n_i)))

    theta_n_values_complex = np.array(theta_n_values_complex)

    # Initialize an empty list to store diagonal matrices
    diagonal_matrices = []

    # Transform each row into a diagonal matrix
    for row in theta_n_values_complex:
        diagonal_matrix = np.diag(row[:Ns])
        diagonal_matrices.append(diagonal_matrix)

    # Convert diagonal_matrices to a NumPy array
    diagonal_matrices = np.array(diagonal_matrices)
    # print(np.shape(diagonal_matrices))

    results_list = []

    for row_index in range(diagonal_matrices.shape[0]):
        # Get the corresponding diagonal matrix for the current row/user is 1x1
        single_row_diag = diagonal_matrices[row_index]

        # Calculate the starting and ending indices for slicing based on row_index
        start_index = row_index * inc
        end_index = start_index + inc

        # Extract the single column from h_rk_transpose using slicing and transpose
        single_row = h_rk_transpose[row_index, start_index:end_index]
        
        # Reshape the single_row to (1, inc)
        single_row = single_row.reshape(1, inc)
        
        # Perform the dot product between f_m_transpose (1, inc) and the current diagonal matrix (inc, inc)
        result_inter = np.dot(single_row, single_row_diag)

        # Perform the final matrix multiplication of result_inter (1, inc) and a subset of G (inc, 1)
        subset_G = G[start_index:end_index]
        result = np.dot(result_inter, subset_G)
        results_list.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results_list)
    results_array = results_array.reshape(Nt, K)

    return results_array

def theta_matrix_ideal(continuous, h_dk, h_rk, G_1, K, Ns, L):
    theta = np.zeros((K, Ns, Ns), dtype=np.complex128)
    for m in range(K):
        if L == 1:
            g_face = G_1[0, :]  # Select channels from the single face
        else:
            chosen_face = m % L  # Sequentially choose IRS face for each user
            g_face = G_1[chosen_face, :]  # Select channels from the chosen face

        theta_n = wrapToPi(np.angle(h_dk[m]) - (np.angle(h_rk[:, m]) + np.angle(g_face)))
        phi_complex = 1 * np.exp(1j * theta_n)

        row_val = phi_complex
        for n in range(Ns):
            theta[m, n, n] = row_val[n]

    return theta

def theta_matrix_practical(continuous, h_dk, h_rk, g, K, Ns, B_min, phi, a, quantized_theta_set):
    '''
        Computes the phase shifts performed by each IRS element.
        Inputs:
            continuous = True if phase shifts are modelled as continuous (-pi to pi)
            h_dk = Direct link from BS to user, if input as None, not considered.
            h_rk = Indirect link from IRS to User of shape (Ns,K)
            g = Fading channel from BS to IRS of shape (Ns, 1)
            K = Num of Users
            Ns = Num of IRS elements 
            B_min = Mininum value of B for quantization
            phi, a = Parameter for practical phase shifts
            quantized_theta_set = Quantization according to quantization bit
        Return:
            Returns theta diagnol matrix, containing practical phase shifts wrt each IRS element. Shape (K,Ns,Ns)
    '''
    inc = int(Ns / K)
    B = np.zeros((K,inc))
    v = np.zeros((K,inc),dtype=np.complex128)
    theta_n = np.zeros((K, inc), dtype=complex)
    nearest_quantized_theta = np.zeros((K, inc), dtype=complex)

    if(continuous == True and quantized_theta_set == None):
            for m in range(K):
                for n in range(inc):
                    theta_n[m] = wrapToPi((np.angle(h_dk[m])) - (np.angle(h_rk[m*inc:(m+1)*inc, m]) + np.angle(g[m*inc:(m+1)*inc, 0])))
                    B[m] = (1 - B_min) * ((np.sin(theta_n[m] - phi) + 1)/2)**a + B_min
                    v[m] = B[m] * np.exp(1j*theta_n[m])

    else:
            for m in range(K):
                for n in range(inc):
                    theta_n[m] = wrapToPi((np.angle(h_dk[m])) - (np.angle(h_rk[m*inc:(m+1)*inc, m]) + np.angle(g[m*inc:(m+1)*inc, 0])))
                    nearest_quantized_theta[m][n] = quantized_theta_set[np.argmin(np.abs(theta_n[m][n] - quantized_theta_set))]
                    B[m] = ((1 - B_min) * ((np.sin(nearest_quantized_theta[m] - phi) + 1) / 2) ** a + B_min)
                    v[m] = B[m] * np.exp(1j*nearest_quantized_theta[m])

    theta = np.zeros((K,inc,inc), dtype= np.complex128)
    row_val = []
    for m in range(K):
        row_val = v[m,:]
        for n in range(inc):
            theta[m,n,n] = row_val[n]
    return theta

def prod_matrix(theta, h_rk_h, G_1, K, Ns, L, closest_face_indices):
    prod_fgtheta = np.zeros((K, 1), dtype=np.complex128)
    for m in range(K):
        if L == 1:
            g_face = G_1[0, :]  # Select channels from the single face
        else:
            chosen_face = closest_face_indices[m]  # Choose the closest IRS face for each user
            g_face = G_1[chosen_face, :]  # Select channels from the chosen face

        prod_f_theta = np.matmul(h_rk_h[m, :], theta[m, :, :])  # Multiply h_rk_h with theta
        prod_fgtheta[m] = np.matmul(prod_f_theta, g_face[:, np.newaxis])  # Multiply with the chosen face's channels

    return prod_fgtheta

def compute_power_at_base_station(K, vn, Pt, PB_dBm):
    # Convert PB from dBm to watts
    PB_watts = dBm2pow(PB_dBm)

    # Calculate P1
    P1 = K*(Pt/vn) + PB_watts

    return P1

def compute_power_consumption_at_ris(K, B, Ns):
    # Define power consumption levels for different quantization bits
    # if B == 1:
    #     power_per_element = 5
    # elif B == 2:
    #     power_per_element = 10
    # elif B == 3:
    #     power_per_element = 15
    if B == None:
        power_per_element = 0.5e-3  # Default power consumption for continuous case

    # Calculate total power consumption for all Ns elements
    power_consumption = power_per_element 
    # power_consumption = (10**(power_consumption/10))/1000
    total_power_consumption = power_consumption * Ns * K
    return total_power_consumption

def compute_ue_power(K, Pu_dBm):
    # Convert Pu from dBm to watts
    Pu_watts = dBm2pow(Pu_dBm)

    # Calculate P1
    P1 = Pu_watts * K

    return P1

def compute_sw_power(P_sw_dBm):
    # Convert Pu from dBm to watts
    P_sw_watts = dBm2pow(P_sw_dBm)
    return P_sw_watts

def compute_area(GRID_RADIUS):
    area = np.pi * (GRID_RADIUS)**2
    return area

# def calculate_values_for_radius(GRID_RADIUS, K):
#     grid_area = compute_area(GRID_RADIUS)
#     Threshold = GRID_RADIUS / 10 # Changed the factor from 2 to 10

#     IRS_x1 = Threshold*np.cos(0.92729522)
#     IRS_y1 = Threshold*np.sin(0.92729522)

#     IRS_x2 = IRS_x1
#     IRS_y2 = -1 * IRS_y1

#     IRS_POSITION_1 = (IRS_x1, IRS_y1, 10)
#     IRS_POSITION_2 = (IRS_x2, IRS_y2, 10)
    
#     user_positions = generate_user_positions_3D(K, GRID_RADIUS)
#     loc_U = user_positions

#     return grid_area, IRS_POSITION_1, IRS_POSITION_2, loc_U , Threshold

def generate_positions_IRS(GRID_RADIUS):
        IRS_X = np.zeros(4)
        IRS_Y = np.zeros(4)
        IRS_Z = np.zeros(4)
        for i in range(len(GRID_RADIUS)):
                IRS_X[i] = GRID_RADIUS[i]*np.cos(0.92729522)
                IRS_Y[i] = GRID_RADIUS[i]*np.sin(0.92729522)
                IRS_Z[i] = 10
        IRS_POSITIONS_1 = np.column_stack((IRS_X, IRS_Y, IRS_Z))
        IRS_POSITIONS_2 = np.column_stack((IRS_X, -IRS_Y, IRS_Z))
        return IRS_POSITIONS_1, IRS_POSITIONS_2
