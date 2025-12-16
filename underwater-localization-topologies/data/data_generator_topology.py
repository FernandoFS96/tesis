import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
import os
from tqdm import tqdm
import pickle

'''
Use:
    python data_generator_topology.py
'''

def range_m(init, end, step):  # Matlab-like range, including end!
    out = np.arange(init, end, step)
    if out[-1] + step <= end:  # Include end only if it is in the interval
        out = np.append(out, out[-1] + step)
    return out

def power(complex_array):
    return np.sum(np.power(np.absolute(complex_array), 2))

class channel():
    def __init__(self, params=None, number_of_processes=-1, load=False, name='0', topology='ellipsoidal', precomputed_trajectories=None):
        # params: channel parameters (default if None)
        # number of processes: for parallelization (1 for single, -1 for as much as possible)
        # topology: sensor arrangement ('ellipsoidal', 'random', 'aligned')
        # precomputed_trajectories: trajectories to reuse across topologies
        
        self.h = None  # To store the impulse response
        self.traj = None  # To store trajectories
        self.nop = number_of_processes
        self.topology = topology
        self.precomputed_trajectories = precomputed_trajectories

        if self.nop == -1:
            self.nop = cpu_count()

        self.params = params
        if self.params is None:
            print(f'\nLoading channel params: default used (topology: {topology})')
            self.default_params()
        else:
            print(f'\nLoading channel params: non default used (topology: {topology})')
        
        if load:
            try:
                print(f'Loading channel matrix for topology: {topology}')
                base_dir = 'data'
                topology_dir = f'{base_dir}/channel_option_{name}/{topology}'
                self.h = np.load(f'{topology_dir}/channel_info/channel_h_{name}.npy')
                self.traj = np.load(f'{topology_dir}/channel_info/trajs_{name}.npy')
                self.r_posicion = np.load(f'{topology_dir}/channel_info/sensor_positions_{name}.npy')
            except:
                print(f'Obtaining channel matrix for topology: {topology}')
                self.h = self.obtain_h().astype(np.complex64)
                self.save_channel_info(name)
        else:
            print(f'Obtaining channel matrix for topology: {topology}')
            self.h = self.obtain_h().astype(np.complex64)
            self.save_channel_info(name)

    def save_channel_info(self, name):
        base_dir = 'data'
        topology_dir = f'{base_dir}/channel_option_{name}/{self.topology}'
        info_dir = f'{topology_dir}/channel_info'
        os.makedirs(info_dir, exist_ok=True)

        # Save channel data for this topology
        np.save(f'{info_dir}/channel_h_{name}.npy', self.h)
        np.save(f'{info_dir}/trajs_{name}.npy', self.traj)
        np.save(f'{info_dir}/sensor_positions_{name}.npy', self.r_posicion)

    def default_params(self):
        channel_info = {'h0': 50,  # Surface height(depth)[m]
                        'ht0': 50,  # TX height [m]
                        'hr0': 1,  # RX height [m]
                        'd0': 500,  # channel distance [m]
                        'k': 1.700000,  # spreading_factor
                        'c': 1500.000000,  # speed_of_sound_in_water_[m / s]
                        'c2': 1200.000000,  # speed_of_sound_in_bottom_[m / s]
                        'cut': 50.000000,  # minimum_relative_path_strength
                        'fmin': 10000.000000,  # minimum_frequency_[Hz]
                        'B': 10000.000000,  # bandwidth_[Hz]
                        'df': 25.000000,  # frequency_resolution_[Hz]
                        'dt': 6.045000,  # time_resolution_[seconds]
                        'T_SS': 6.000000,  # coherence_time_of_the_small - scale_variations_[seconds]
                        'sig2s': 0.0,  # 1.125000, # variance_of_S_S_surface_variations_[m ^ 2]
                        'sig2b': 0.0,  # 0.562500, # variance_of_S_S_bottom_variations_[m ^ 2]
                        'B_delp': 0.000500,  # 3 - dB_width_of_the_p.s.d._of_intra - path_delays_[Hz]
                        'Sp': 0,  # 20, # number_of_intra - paths
                        'mu_p': 0.0,  # 0.025000, # mean_of_intra - path_amplitudes
                        'nu_p': 0.0,  # 0.000001, # variance_of_intra - path_amplitudes
                        'T_tot': 6.000000,  # total_duration_of_the_simulated_signal_[seconds]
                        'h_bnd': [0, 0],  # [-1.0, 1.0], # Range of surface height
                        'ht_bnd': [0, 0],  # [-1.0, 1.0], # Range of tx height
                        'hr_bnd': [0, 0],  # [-1.0, 1.0], # Range of rx height
                        'd_bnd': [0, 0],  # [-10.0, 10.0], # Range of channel distance
                        'sig_h': 0.0,  # 1.000000, #L_S_standard_deviation_of_surface_height_[m]
                        'sig_ht': 0.0,  # 1.000000, # L_S_standard_deviation_of_transmitter_height_[m]
                        'sig_hr': 0.0,  # 1.000000, # L_S_standard_deviation_of_receiver_height_[m]
                        'sig_d': 0.0,  # 1, # L_S_standard_deviation_of_receiver_height_[m]
                        'a_AR': 0.900000,  # AR_parameter_for_generating_L_S_variations
                        }

        channel_info['fmax'] = channel_info['fmin'] + channel_info['B']
        channel_info['N_LS'] = int(np.round(channel_info['T_tot'] / channel_info['T_SS']))

        dopp_params = [0.197777, 0.797882, 0.100000, 5.738910, 0.318765, 0.000000, 0.000000, 0.000000, 0.050000,
                       0.010000]

        self.params = {'ci': channel_info,
                       'Dopp_params': dopp_params,
                       'n_sensors': 10,  # Number of sensors
                       'radius_r': 1000,  # To obtain sensor position
                       'n_traj': 150,  # Number of trajectories
                       'ppt': 50,  # Number of points per trajectory
                       'm': 0.7,  # Modulation index
                       'T': 20,  # Sampling period
                       }
        self.params['w0'] = np.pi / self.params['T']  # Rotation pulsation

    def generate_sensor_positions(self):
        """Generate sensor positions based on topology"""
        n_sensors = self.params['n_sensors']
        radius_r = self.params['radius_r']
        
        if self.topology == 'ellipsoidal':
            # Original implementation
            x_r = radius_r * np.cos(2 * np.pi / n_sensors * 
                                   np.linspace(0, n_sensors, n_sensors + 1))
            y_r = radius_r / 2 * np.sin(2 * np.pi / n_sensors * 
                                       np.linspace(0, n_sensors, n_sensors + 1))
            r_posicion = np.zeros([3, n_sensors])
            r_posicion[0, :] = x_r[0:n_sensors]
            r_posicion[1, :] = y_r[0:n_sensors]
            r_posicion[2, :] = self.params['ci']['hr0']
            
        elif self.topology == 'random':
            # Random scatter in the plane
            np.random.seed(18)  # Fixed seed for reproducibility
            # Generate random positions within a square region
            max_range = radius_r * 1.5  # Extend the range a bit for better coverage
            r_posicion = np.zeros([3, n_sensors])
            r_posicion[0, :] = np.random.uniform(-max_range, max_range, n_sensors)
            r_posicion[1, :] = np.random.uniform(-max_range/2, max_range/2, n_sensors)
            r_posicion[2, :] = self.params['ci']['hr0']
            
        elif self.topology == 'aligned':
            # Aligned along x-axis
            r_posicion = np.zeros([3, n_sensors])
            # Spread sensors evenly along x-axis
            r_posicion[0, :] = np.linspace(-radius_r * 1.5, radius_r * 1.5, n_sensors)
            r_posicion[1, :] = 0  # All sensors at y=0
            r_posicion[2, :] = self.params['ci']['hr0']
            
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
        
        return r_posicion

    def generate_trajectories(self):
        """Generate trajectories or use precomputed ones"""
        if self.precomputed_trajectories is not None:
            return self.precomputed_trajectories
        
        # Generate new trajectories
        n_traj = self.params['n_traj']
        ppt = self.params['ppt']
        
        radio_t = 100 + 1000 * np.random.rand(n_traj)
        fase0 = 2 * np.pi * np.random.rand(n_traj)
        omega0 = 2 * np.pi * np.random.rand(n_traj)

        traj = np.zeros([3, n_traj, ppt + 1])
        aux_1 = np.linspace(0, ppt, ppt + 1)
        aux_2 = np.linspace(ppt, 2 * ppt, ppt + 1)
        
        for it in range(n_traj):
            traj[0, it, :] = radio_t[it] / ppt * aux_2 * np.cos(
                omega0[it] / ppt * aux_1 + fase0[it])
            traj[1, it, :] = radio_t[it] / ppt * aux_2 * np.sin(
                omega0[it] / ppt * aux_1 + fase0[it])
            traj[2, it, :] = 0
        
        return traj

    def obtain_h(self):
        # Generate sensor positions based on topology
        self.r_posicion = self.generate_sensor_positions()
        
        # Generate or use precomputed trajectories
        self.traj = self.generate_trajectories()
        
        # Process values for each pair (sensor, trajectory)
        def process_sensor(ise):
            import scipy.signal as signal
            Lf = len(range_m(self.params['ci']['fmin'], self.params['ci']['fmax'], self.params['ci']['df']))
            h_val = np.zeros([self.params['n_traj'], self.params['ppt'], Lf],
                             dtype=np.complex128)
            
            for itx in tqdm(range(self.params['n_traj']), 
                          desc=f"Sensor {ise+1} processing trajectories ({self.topology})", 
                          leave=False):
                for ptx in range(self.params['ppt']):
                    # [Rest of the processing code remains the same as original]
                    d0 = np.linalg.norm(self.r_posicion[:, ise] - self.traj[:, itx, ptx])
                    
                    # Obtain frequency parameters
                    f_vec = range_m(self.params['ci']['fmin'], self.params['ci']['fmax'], self.params['ci']['df'])
                    Lf = len(f_vec)
                    fc = (self.params['ci']['fmin'] + self.params['ci']['fmax']) / 2
                    f0 = self.params['ci']['fmin']   
                    # Obtain rest of channel parameters
                    t_vec = range_m(0, self.params['ci']['T_SS'], self.params['ci']['dt'])
                    Lt = len(t_vec)
                    t_tot_vec = range_m(0, self.params['ci']['T_tot'], self.params['ci']['dt'])
                    Lt_tot = len(t_tot_vec)
                    # Obtain doppler parameters
                    Dopp_params = np.reshape(self.params['Dopp_params'], [Lt_tot, 10])
                    # Doppler drift
                    vtd_tot = Dopp_params[:, 0]
                    theta_td_tot = Dopp_params[:, 1]
                    vrd_tot = Dopp_params[:, 2]
                    theta_rd_tot = Dopp_params[:, 3]
                    # vehicular
                    v_t0 = (self.traj[:, itx, ptx + 1] - self.traj[:, itx, ptx]) / self.params['ci']['T_tot']
                    v_t0_proy = np.sum(v_t0 * (self.r_posicion[:, ise] - self.traj[:, itx, ptx])) / d0
                    vtv_tot = v_t0_proy + Dopp_params[:, 4]
                    theta_tv_tot = Dopp_params[:, 5]
                    vrv_tot = Dopp_params[:, 6]
                    theta_rv_tot = Dopp_params[:, 7]
                    # surface
                    Aw_tot = Dopp_params[:, 8]
                    fw_tot = Dopp_params[:, 9]

                    # Large-scale loop
                    H_LS = np.zeros([Lf, int(Lt * self.params['ci']['N_LS'])], dtype=np.complex128)
                    del_h = 0
                    del_ht = 0
                    del_hr = 0
                    del_d = 0
                    h = self.params['ci']['h0']
                    ht = self.params['ci']['ht0']
                    hr = self.params['ci']['hr0']
                    d = d0
                    adopp0 = np.zeros(50)

                    for LScount in range(self.params['ci']['N_LS']):
                        rndvec = np.random.randn(4)
                        del_h = self.params['ci']['a_AR'] * del_h \
                                + np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_h'] * rndvec[0]
                        if del_h > self.params['ci']['h_bnd'][1] or del_h < self.params['ci']['h_bnd'][0]:
                            del_h = del_h \
                                    - 2 * np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_h'] * \
                                    rndvec[0]
                        htemp = h
                        h = self.params['ci']['h0'] + del_h

                        del_ht = self.params['ci']['a_AR'] * del_ht \
                                 + np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_ht'] * rndvec[1]
                        if del_ht > self.params['ci']['ht_bnd'][1] or del_ht < self.params['ci']['ht_bnd'][0]:
                            del_ht = del_ht \
                                     - 2 * np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_ht'] * \
                                     rndvec[1]

                        httemp = ht
                        ht = self.params['ci']['ht0'] + del_ht

                        del_hr = self.params['ci']['a_AR'] * del_hr \
                                 + np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_hr'] * rndvec[2]
                        if del_hr > self.params['ci']['hr_bnd'][1] or del_hr < self.params['ci']['hr_bnd'][0]:
                            del_hr = del_hr \
                                     - 2 * np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_hr'] * \
                                     rndvec[2]

                        hrtemp = hr
                        hr = self.params['ci']['hr0'] + del_hr

                        del_d = self.params['ci']['a_AR'] * del_d \
                                + np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_d'] * rndvec[3]
                        if del_d > self.params['ci']['d_bnd'][1] or del_d < self.params['ci']['d_bnd'][0]:
                            del_d = del_d \
                                    - 2 * np.sqrt(1 - self.params['ci']['a_AR'] ** 2) * self.params['ci']['sig_d'] * \
                                    rndvec[3]

                        dtemp = d
                        d += del_d

                        def absorption(f):
                            alpha = 0.11 * np.power(f, 2) / (1 + np.power(f, 2)) \
                                    + 44 * np.power(f, 2) / (4100 + np.power(f, 2)) \
                                    + 2.75 * 10 ** (-4) * np.power(f, 2) \
                                    + 0.003
                            indvlf = f < 0.3
                            alphas = 2 * 10 ** (-3)
                            alpha[indvlf] = alphas + 0.11 * np.power(f[indvlf], 2) / (1 + np.power(f[indvlf], 2)) \
                                            + 0.011 * np.power(f[indvlf], 2)
                            return alpha

                        def reflcoeff(theta, c1, c2):
                            rho1 = 1000
                            rho2 = 1800
                            x1 = rho2 / c1 * np.sin(theta)
                            x2 = rho1 / c2 * np.sqrt(1 - (c2 / c1) ** 2 * np.cos(theta) ** 2)
                            thetac = np.real(np.arccos(c1 / c2 + 0 * 1j))
                            if theta < thetac:
                                if thetac == 0:
                                    refl = -1
                                else:
                                    refl = np.exp(1j * np.pi * (1 - theta / thetac))
                            if theta >= thetac:
                                refl = (x1 - x2) / (x1 + x2)
                            return refl

                        def mpgeometry(h, ht, hr, d, f, k, cut, c, c2):
                            f = np.array([f]).astype(float)
                            a = np.power(10.0, absorption(f / 1000) / 10)
                            a = np.power(a, 1 / 1000)
                            nr = 0

                            theta = np.array([np.arctan((ht - hr) / d)])
                            l = np.array([np.sqrt((ht - hr) ** 2 + d ** 2)])
                            dell = np.array([l[0] / c])
                            A = np.array([l[0] ** k * np.power(a, l[0])])
                            ns = np.array([0])
                            nb = np.array([0])
                            G = np.array([1 / np.sqrt(A[0])])
                            Gamma = np.array([1])
                            hp = np.array([1])
                            path = np.array([0], dtype=int)
                            tau = np.array([0])

                            while min(abs(G)) >= G[0] / cut:
                                nr = nr + 1
                                for case in range(2):
                                    if case == 0:
                                        p = 2 * nr - 1
                                    else:
                                        p = 2 * nr
                                        path = np.logical_not(path).astype(int)

                                    first = path[0]
                                    last = path[-1]
                                    nb = np.append(nb, np.sum(path))
                                    ns = np.append(ns, nr - nb[p])
                                    heff = (1 - first) * ht + first * (h - ht) + (nr - 1) * h + (
                                                1 - last) * hr + last * (h - hr)
                                    l = np.append(l, np.sqrt(heff ** 2 + d ** 2))
                                    theta = np.append(theta, np.arctan(heff / d))
                                    if first == 1:
                                        theta[p] = - theta[p]
                                    dell = np.append(dell, l[p] / c)
                                    tau = np.append(tau, dell[p] - dell[0])
                                    A = np.append(A, (l[p] ** k) * (np.power(a, l[p])))
                                    Gamma = np.append(Gamma,
                                                      reflcoeff(np.abs(theta[p]), c, c2) ** nb[p] * (-1) ** ns[p])
                                    G = np.append(G, Gamma[p] / np.sqrt(A[p]))
                                    hp = np.append(hp, Gamma[p] / np.sqrt(((l[p] / l[0]) ** k) * (a ** (l[p] - l[0]))))

                                path = np.append(path, np.logical_not(path[-1]).astype(int))

                            return l, tau, Gamma, theta, ns, nb, hp
                        # FIND LARGE SCALE MODEL PARAMETERS
                        lmean, taumean, Gamma, theta, ns, nb, hp = mpgeometry(h, h - ht, h - hr, d, fc,
                                                                              self.params['ci']['k'],
                                                                              self.params['ci']['cut'],
                                                                              self.params['ci']['c'],
                                                                              self.params['ci']['c2'])
                        # ignore paths with delays longer than allowed by frequency resolution:
                        lmean = lmean[taumean < 1 / self.params['ci']['df']]
                        theta = theta[taumean < 1 / self.params['ci']['df']]
                        ns = ns[taumean < 1 / self.params['ci']['df']]
                        nb = nb[taumean < 1 / self.params['ci']['df']]
                        hp = hp[taumean < 1 / self.params['ci']['df']]
                        taumean = taumean[taumean < 1 / self.params['ci']['df']]
                        P = len(lmean) # Number of paths
                        # Reference path transfer function
                        H0 = 1 / np.sqrt(np.power(lmean[0], self.params['ci']['k'])
                                         * np.power(np.power(10.0, absorption(f_vec / 1000) / 10000), lmean[0]))
                        H = hp[0] * np.tile(np.exp(-1j * 2 * np.pi * f_vec * taumean[0]), [1, Lt])
                        # Find doppler rates:
                        sig_delp = np.sqrt(1 / self.params['ci']['c'] ** 2 * np.power(2 * np.sin(theta), 2)
                                           * (ns * self.params['ci']['sig2s'] + nb * self.params['ci']['sig2b']))
                        # drifting:
                        vtd = vtd_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        theta_td = theta_td_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        vrd = vrd_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        theta_rd = theta_rd_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        # vehicular:
                        vtv = vtv_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        theta_tv = theta_tv_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        vrv = vrv_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        theta_rv = theta_rv_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        # surface:
                        Aw = Aw_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        fw = fw_tot[1 + (LScount - 1) * (Lt - 1) - 1: 1 + LScount * (Lt - 1)]
                        vw = 2 * np.pi * fw * Aw
                        # First path doppler
                        vdrift = vtd * np.cos(theta[0] - theta_td) - vrd * np.cos(theta[0] + theta_rd)
                        adrift = vdrift / self.params['ci']['c']
                        vvhcl = 0
                        avhcl = vvhcl / self.params['ci']['c']
                        vsurf = 0
                        asurf = vsurf / self.params['ci']['c']

                        adopp = adrift + avhcl + asurf * ns[0]
                        eff_adopp = adopp0[0] + np.cumsum(adopp)
                        Dopp = np.exp(1j * 2 * np.pi * f_vec * (eff_adopp * self.params['ci']['dt']))
                        adopp0[0] = eff_adopp[-1]
                        H = H * Dopp
                        # small - scale simulation: Direct method
                        for p in range(1, P):
                            gamma = np.zeros([Lf, Lt])
                            for counti in range(self.params['ci']['Sp']):
                                gamma_pi = self.params['ci']['mu_p'] + self.params['ci']['nu_p'] * np.random.randn(Lt)
                                gamma_pi = np.tile(gamma_pi, [Lf, 1]) * self.params['ci']['Sp']
                                deltau_pi = np.zeros([Lf, Lt])
                                w_delpi = sig_delp[p] \
                                          * np.sqrt(1 - np.exp(-1 * np.pi * self.params['ci']['B_delp']
                                                               * self.params['ci']['dt']) ** 2) * np.random.randn(
                                    2 * Lt)

                                temp_deltau_pi = signal.lfilter([1], [1, -np.exp(-np.pi * self.params['ci']['B_delp']
                                                                                 * self.params['ci']['dt'])], w_delpi)
                                for countf in range(Lf):
                                    deltau_pi[countf, :] = temp_deltau_pi[Lt:]
                                gamma = gamma + gamma_pi * np.exp(
                                    -1j * 2 * np.pi * np.tile(f_vec, [1, Lt]).T * deltau_pi)
                            # Doppler term:
                            vdrift = vtd * np.cos(theta[p] - theta_td) - vrd * np.cos(theta[p] + theta_rd)
                            adrift = vdrift / self.params['ci']['c']
                            vvhcl = vtv * np.cos(theta[p] - theta_tv) - vrv * np.cos(theta[p] + theta_rv) - (
                                    vtv * np.cos(theta[0] - theta_tv) - vrv * np.cos(theta[0] + theta_rv))
                            avhcl = vvhcl / self.params['ci']['c']

                            phi_pj = 2 * np.pi * np.random.rand(ns[p]) - np.pi
                            sum_j = np.zeros(Lt)
                            for jcount in range(ns[p]):
                                sum_j = sum_j + np.sin(phi_pj[jcount] + 2 * np.pi * fw * t_vec)
                            vsurf = 2 * vw * np.sin(theta[p]) * sum_j
                            asurf = vsurf / self.params['ci']['c']

                            adopp = adrift + avhcl + asurf * ns[p]
                            eff_adopp = adopp0[p] + np.cumsum(adopp)
                            Dopp = np.exp(1j * 2 * np.pi * f_vec * eff_adopp * self.params['ci']['dt'])
                            adopp0[p] = eff_adopp[-1]
                            # Multiply gamma by hp:
                            gamma = np.squeeze(gamma) * Dopp
                            H = H + hp[p] * np.tile(np.exp(-1j * 2 * np.pi * f_vec * taumean[p]), [1, Lt]) * gamma
                        H = np.tile(H0, [1, Lt]) * H
                        H_LS[:, LScount * Lt: (LScount + 1) * Lt] = H.T
                    # find channel impulse response:
                    Lt_tot = np.shape(H_LS)[1]
                    hmat = np.zeros([Lf, Lt_tot], dtype=np.complex128)
                    for countt in range(Lt_tot):
                        hmat[:, countt] = np.fft.ifft(H_LS[:, countt])

                    h_val[itx, ptx, :] = np.squeeze(hmat)
            return h_val

        out = Parallel(n_jobs=self.nop, verbose=0) \
            (delayed(process_sensor)(ise=ise) for ise in range(self.params['n_sensors']))

        return np.array(out).T  # tau x t x traj x sensor

    def filter(self, n, snr=0, nt=10, multiprocessing=True, specific=None, signal_type='sinusoid', rep=1):
        # [Keep the original filter method unchanged]
        print(f'Filtering for topology: {self.topology}...')
        # Obtain trajs
        if specific is None:
            if nt > self.params['n_traj']:
                trjs = np.random.choice(self.params['n_traj'], nt, replace=True).tolist()
            else:
                trjs = np.random.choice(self.params['n_traj'], nt, replace=False).tolist()
        else:
            trjs = specific
        assert nt == len(trjs)
        trjs = np.repeat(trjs, rep)

        def process_sensor(ise):
            h = np.roll(self.h, 50, axis=0)
            s = np.zeros([n, self.params['ppt']])
            x = np.zeros([n, self.params['ppt']])

            def conv(u, v):
                npad = len(v) - 1
                u_padded = np.pad(u, (npad // 2, npad - npad // 2), mode='constant')
                return np.convolve(u_padded, v, 'valid')

            for ipt in range(self.params['ppt']):
                if signal_type == 'sinusoid':
                    x[:, ipt] = np.real(np.exp(1j * self.params['w0'] * np.arange(n)))
                elif signal_type == 'sinusoid_cav':
                    s[:, ipt] = conv(np.exp(-10 * np.arange(n) / n), np.random.randn(2 * n - 1))
                    x[:, ipt] = (1 + self.params['m'] * np.cos(self.params['w0'] * np.arange(n))) * s[:, ipt]
                    x[:, ipt] = x[:, ipt] / np.sqrt(np.sum(np.power(x[:, ipt], 2)))
                else:
                    raise RuntimeError('Signal type not recognized')

            y_o = np.zeros([h.shape[0], h.shape[1], len(trjs)])
            i = 0
            for itx in trjs:
                if n < 2 * h.shape[0]:
                    raise RuntimeError('Too low number of samples')
                y = np.zeros([h.shape[0], self.params['ppt']])
                n_aux = np.random.randn(h.shape[0], self.params['ppt'])
                for ptx in range(self.params['ppt']):
                    signal = np.real(np.convolve(x[:, ptx], h[:, ptx, itx, ise], 'valid'))[0: h.shape[0]]
                    noise = n_aux[:, ptx]
                    signal_power = np.sum(np.power(signal, 2))
                    noise_power = np.sum(np.power(noise, 2))
                    noise = noise * np.sqrt(np.power(10, - snr / 10) * signal_power / noise_power)
                    y[:, ptx] = signal + noise
                y_o[:, :, i] = y[:, :]
                i += 1
            return y_o

        if multiprocessing:
            out = Parallel(n_jobs=self.nop, verbose=0) \
                (delayed(process_sensor)(ise=ise) for ise in range(self.params['n_sensors']))
        else:
            out = [process_sensor(ise) for ise in range(self.params['n_sensors'])]

        y_out = np.zeros([self.h.shape[0], self.h.shape[1], len(trjs), self.h.shape[3]])
        for ise in range(self.params['n_sensors']):
            y_out[:, :, :, ise] = out[ise]
        return y_out, self.traj[:, trjs, 0: self.params['ppt']]


def generate_batch_of_trajs(channel, signal_type, n=1024, snr=0, rep=1):
    data, trjs = channel.filter(n, snr=snr, nt=channel.params['n_traj'], signal_type=signal_type, rep=rep)
    return data, trjs


def generate_params(options=None):
    # [Keep the original generate_params function unchanged]
    channel_info = {'h0': 50,  # Surface height(depth)[m]
                    'ht0': 50,  # TX height [m]
                    'hr0': 1,  # RX height [m]
                    'd0': 500,  # channel distance [m]
                    'k': 1.700000,  # spreading_factor
                    'c': 1500.000000,  # speed_of_sound_in_water_[m / s]
                    'c2': 1200.000000,  # speed_of_sound_in_bottom_[m / s]
                    'cut': 50.000000,  # minimum_relative_path_strength
                    'fmin': 10000.000000,  # minimum_frequency_[Hz]
                    'B': 10000.000000,  # bandwidth_[Hz]
                    'df': 25.000000,  # frequency_resolution_[Hz]
                    'dt': 6.045000,  # time_resolution_[seconds]
                    'T_SS': 6.000000,  # coherence_time_of_the_small - scale_variations_[seconds]
                    'sig2s': 1.125000,  # variance_of_S_S_surface_variations_[m ^ 2]
                    'sig2b': 0.562500,  # variance_of_S_S_bottom_variations_[m ^ 2]
                    'B_delp': 0.000500,  # 3 - dB_width_of_the_p.s.d._of_intra - path_delays_[Hz]
                    'Sp': 20,  # number_of_intra - paths
                    'mu_p': 0.025000,  # mean_of_intra - path_amplitudes
                    'nu_p': 0.000001,  # variance_of_intra - path_amplitudes
                    'T_tot': 6.000000,  # total_duration_of_the_simulated_signal_[seconds]
                    'h_bnd': [-1.0, 1.0],  # Range of surface height
                    'ht_bnd': [-1.0, 1.0],  # Range of tx height
                    'hr_bnd': [-1.0, 1.0],  # Range of rx height
                    'd_bnd': [-10.0, 10.0],  # Range of channel distance
                    'sig_h': 1.000000,  # L_S_standard_deviation_of_surface_height_[m]
                    'sig_ht': 1.000000,  # L_S_standard_deviation_of_transmitter_height_[m]
                    'sig_hr': 1.000000,  # L_S_standard_deviation_of_receiver_height_[m]
                    'sig_d': 1.000000,  # L_S_standard_deviation_of_receiver_height_[m]
                    'a_AR': 0.900000,  # AR_parameter_for_generating_L_S_variations
                    }
    if isinstance(options, str):
        if options == 'no_var':
            aux = 0.0
    else:
        aux = options
    
    channel_info['sig2s'] *= aux
    channel_info['sig2b'] *= aux
    channel_info['mu_p'] *= aux
    channel_info['nu_p'] *= aux
    channel_info['Sp'] = int(channel_info['Sp'] * aux)
    channel_info['h_bnd'] = [channel_info['h_bnd'][0] * aux, channel_info['h_bnd'][1] * aux]
    channel_info['ht_bnd'] = [channel_info['ht_bnd'][0] * aux, channel_info['ht_bnd'][1] * aux]
    channel_info['hr_bnd'] = [channel_info['ht_bnd'][0] * aux, channel_info['ht_bnd'][1] * aux]
    channel_info['d_bnd'] = [channel_info['ht_bnd'][0] * aux, channel_info['ht_bnd'][1] * aux]
    channel_info['sig_h'] *= aux
    channel_info['sig_ht'] *= aux
    channel_info['sig_hr'] *= aux
    channel_info['sig_d'] *= aux

    channel_info['fmax'] = channel_info['fmin'] + channel_info['B']
    channel_info['N_LS'] = int(np.round(channel_info['T_tot'] / channel_info['T_SS']))

    dopp_params = [0.197777, 0.797882, 0.100000, 5.738910, 0.318765, 0.000000, 0.000000, 0.000000, 0.050000, 0.010000]

    params = {'ci': channel_info,
              'Dopp_params': dopp_params,
              'n_sensors': 10,
              'radius_r': 1000,
              'n_traj': 150,
              'ppt': 50,
              'm': 0.7,
              'T': 20,
              }
    params['w0'] = np.pi / params['T']
    return params

def plot_validation(channels_dict, option_idx=0):
    """Plot one trajectory with all three sensor arrangements for validation"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    trajectory_idx = 5  # Use first trajectory for visualization
    
    topologies = ['ellipsoidal', 'random', 'aligned']
    titles = ['Ellipsoidal Topology', 'Random Topology', 'Aligned Topology']
    
    # Calculate global limits for all plots
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    
    # First pass to find global limits
    for topology in topologies:
        c = channels_dict[topology]
        traj = c.traj[:, trajectory_idx, :]
        sensors = c.r_posicion
        
        x_min = min(x_min, traj[0, :].min(), sensors[0, :].min())
        x_max = max(x_max, traj[0, :].max(), sensors[0, :].max())
        y_min = min(y_min, traj[1, :].min(), sensors[1, :].min())
        y_max = max(y_max, traj[1, :].max(), sensors[1, :].max())
    
    # Add some padding to the limits (10% of the range)
    x_padding = 0.1 * (x_max - x_min)
    y_padding = 0.1 * (y_max - y_min)
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    for idx, (topology, title) in enumerate(zip(topologies, titles)):
        ax = axes[idx]
        c = channels_dict[topology]
        
        # Plot trajectory
        traj = c.traj[:, trajectory_idx, :]
        ax.plot(traj[0, :], traj[1, :], 'b-', linewidth=2, alpha=0.7)
        ax.plot(traj[0, 0], traj[1, 0], 'go', markersize=10)
        ax.plot(traj[0, -1], traj[1, -1], 'ko', markersize=10)
        
        # Plot sensors
        sensors = c.r_posicion
        ax.scatter(sensors[0, :], sensors[1, :], c='red', s=100, marker='o',
                  edgecolors='black', linewidth=1, zorder=5)
        
        # Add sensor numbers
        for i in range(sensors.shape[1]):
            ax.annotate(f'{i}', (sensors[0, i], sensors[1, i]), 
                       textcoords="offset points", xytext=(0,5), ha='center', fontsize=14)
        
        # Increase font size for labels and ticks
        ax.set_xlabel('X [m]', fontsize=16)
        ax.set_ylabel('Y [m]', fontsize=16)
        ax.set_title(title, fontsize=18, pad=10)
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        ax.grid(True, alpha=0.3)
        
        # Set the same limits for all plots
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_aspect('equal', adjustable='box')
        
        # Only add legend to the first plot
        if idx == 0:
            ax.plot([], [], 'b-', linewidth=2, label='Trajectory')
            ax.plot([], [], 'go', markersize=10, label='Start')
            ax.plot([], [], 'ko', markersize=10, label='End')
            ax.plot([], [], 'ro', markeredgecolor='black', markersize=10, label='Sensors')
            ax.legend(loc='upper left', fontsize=14)  
    #plt.suptitle(f'Validation Plot - Trajectory with Different Sensor Topologies\n(Channel Option: {option_idx})', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    base_dir = 'data'
    plot_dir = f'{base_dir}/validation_plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/topology_comparison_option_{option_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Validation plot saved to {plot_dir}/topology_comparison_option_{option_idx}.png")

def process(channel_options, snr, rep, nop=-1):
    """Process data for all topologies using the same trajectories"""
    topologies = ['ellipsoidal', 'random', 'aligned']
    
    for option_idx, option in enumerate(tqdm(channel_options, desc="Processing channel options")):
        print(f"\n--- Processing Channel Option: {option} ---")
        
        # Generate parameters for this option
        params = generate_params(options=option)
        
        # First, generate trajectories that will be shared across all topologies
        # We create a temporary channel just to generate trajectories
        temp_channel = channel(load=False, params=params, number_of_processes=nop, 
                              name=str(option), topology='ellipsoidal')
        shared_trajectories = temp_channel.traj  # Store the trajectories
        
        channels_for_validation = {}
        
        # Process each topology with the same trajectories
        for topology in topologies:
            print(f"  Processing topology: {topology}")
            
            # Create channel with specific topology and shared trajectories
            c = channel(load=False, params=params, number_of_processes=nop, 
                       name=str(option), topology=topology, 
                       precomputed_trajectories=shared_trajectories)
            
            # Store for validation plotting
            channels_for_validation[topology] = c
            
            # Generate batch of trajectories
            data, trjs = generate_batch_of_trajs(c, 'sinusoid', n=1024, snr=snr, rep=rep)
            
            # Save the generated data
            root_dir = 'data'
            topology_dir = f'{root_dir}/channel_option_{option}/{topology}'
            
            # Create directories for this topology
            os.makedirs(f'{topology_dir}/trajectory', exist_ok=True)
            os.makedirs(f'{topology_dir}/filtered_data', exist_ok=True)
            
            # Save data
            np.save(f'{topology_dir}/trajectory/trajectories.npy', trjs)
            np.save(f'{topology_dir}/filtered_data/filtered_data.npy', data)
            
            print(f" Data saved for topology: {topology}")
        
        # Create validation plot for this option (only for the first option)
        if option_idx == 0:
            plot_validation(channels_for_validation, option)


if __name__ == '__main__':    
    channel_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] #[0.6, 0.7, 0.8, 0.9, 1.0]
    
    snr = 10
    rep = 1
    
    # Set random seed for reproducibility
    np.random.seed(18)
    
    process(channel_options, snr, rep)