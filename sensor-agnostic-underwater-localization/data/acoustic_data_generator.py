import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count
import os
from tqdm import tqdm
import pickle
import argparse
from omegaconf import OmegaConf

'''
Use:
    python acoustic_data_generator.py \
        --n_traj 150 \
        --snr 10 \
        --rep 1

'''

# Path to the data-pipeline config. This is loaded directly (no Hydra) because
# data generation runs as a standalone step.
DEFAULT_TRAJ_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'data_pipeline.yaml')


def load_traj_config(path=None):
    """Load the trajectory-generation config from YAML via OmegaConf."""
    if path is None:
        path = DEFAULT_TRAJ_CONFIG_PATH
    return OmegaConf.load(path)

def range_m(init, end, step):  # Matlab-like range, including end!
    out = np.arange(init, end, step)
    if out[-1] + step <= end:  # Include end only if it is in the interval
        out = np.append(out, out[-1] + step)
    return out

def power(complex_array):
    return np.sum(np.power(np.absolute(complex_array), 2))

class channel():
    def __init__(self, params=None, number_of_processes=-1, load=False, name='0', topology='ellipsoidal', precomputed_trajectories=None, traj_config=None):
        # params: channel parameters (default if None)
        # number of processes: for parallelization (1 for single, -1 for as much as possible)
        # topology: sensor arrangement ('ellipsoidal', 'random', 'aligned')
        # precomputed_trajectories: trajectories to reuse across topologies
        # traj_config: trajectory-generation config (loaded from
        #   config/data_pipeline.yaml if None)

        self.h = None  # To store the impulse response
        self.traj = None  # To store trajectories
        self.nop = number_of_processes
        self.topology = topology
        self.precomputed_trajectories = precomputed_trajectories
        self.traj_config = traj_config if traj_config is not None else load_traj_config()

        if self.nop == -1:
            self.nop = cpu_count()

        self.params = params
        if self.params is None:
            print(f'\nLoading channel params: default used (topology: {topology})')
            self.default_params()
        else:
            print(f'\nLoading channel params: non default used (topology: {topology})')
        
        assert self.params is not None, "params must be initialized"
        
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
        
        assert self.h is not None, "Channel matrix (h) must be initialized"
        assert self.traj is not None, "Trajectories (traj) must be initialized"

    def save_channel_info(self, name):
        base_dir = 'data'
        topology_dir = f'{base_dir}/channel_option_{name}/{self.topology}'
        info_dir = f'{topology_dir}/channel_info'
        os.makedirs(info_dir, exist_ok=True)

        # Save channel data for this topology
        assert self.h is not None, "Channel matrix (h) is None"
        assert self.traj is not None, "Trajectories (traj) is None"
        assert self.r_posicion is not None, "Sensor positions (r_posicion) is None"
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
                       'ppt': 30, #50 # Number of points per trajectory
                       'm': 0.7,  # Modulation index
                       'T': 20,  # Sampling period
                       }
        self.params['w0'] = np.pi / self.params['T']  # Rotation pulsation

    def generate_sensor_positions(self, traj, scale=1.10, min_span=50.0):
        """
        Genera posiciones de sensores adaptadas al tamaño de las trayectorias 'traj'
        para las tres topologías: 'ellipsoidal', 'random' y 'aligned'.

        Parámetros
        ----------
        traj : np.ndarray
            Array (3, n_traj, ppt+1) con las trayectorias ya generadas.
        scale : float
            Factor >1 para dar un pequeño margen alrededor de las trayectorias.
        min_span : float
            Amplitud mínima (m) para evitar rangos demasiado pequeños.

        Devuelve
        --------
        r_posicion : np.ndarray de forma (3, n_sensors)
        """
        assert self.params is not None, "params must be initialized"

        n_sensors = self.params['n_sensors']
        hr0 = self.params['ci']['hr0']

        # --- 1) Medidas robustas del "tamaño" de las trayectorias ---
        xs = traj[0].ravel()
        ys = traj[1].ravel()
        if xs.size == 0:
            raise ValueError("Trajectories array is empty")

        # Caja que contiene las trayectorias con percentiles robustos
        x_lo, x_hi = np.percentile(xs, [2.0, 98.0])
        y_lo, y_hi = np.percentile(ys, [2.0, 98.0])

        # Centro y spans
        cx = 0.5 * (x_lo + x_hi)
        cy = 0.5 * (y_lo + y_hi)
        span_x = max((x_hi - x_lo) * scale, min_span)
        span_y = max((y_hi - y_lo) * scale, min_span)

        # Aseguramos aspecto similar al original de la elipse (b = a/2)
        # Usaremos 'a' a partir del radio robusto r95
        r = np.sqrt(xs**2 + ys**2)
        r95 = float(np.percentile(r, 95)) if r.size > 0 else 1.0
        a = max(scale * r95, min_span / 2)     # semieje mayor ~ tamaño de la espiral
        b = 0.5 * a                             # elipse "aplastada" (como antes)

        # --- 2) Topologías ---
        if self.topology == 'ellipsoidal':
            # Elipse centrada en (cx, cy) con semiejes a y b
            thetas = np.linspace(0.0, 2*np.pi, n_sensors, endpoint=False)
            x = cx + a * np.cos(thetas)
            y = cy + b * np.sin(thetas)

        elif self.topology == 'random':
            # Dispersión uniforme dentro de una caja centrada en (cx, cy)
            # Caja ligeramente rectangular como en tu código original (y la mitad que x)
            max_x = 0.5 * span_x
            max_y = 0.5 * span_y
            # Si quieres mantener la "mitad" en y, puedes forzar max_y = max_x/2
            max_y = max(max_y, min_span / 2.0)
            rng = np.random.default_rng(10)  # reproducibilidad
            x = rng.uniform(cx - max_x, cx + max_x, n_sensors)
            y = rng.uniform(cy - max_y, cy + max_y, n_sensors)

        elif self.topology == 'aligned':
            # Alineados en x, centrados en cx, con y = cy
            # Longitud total ≈ span_x; si quieres menos/mas largo, ajusta factor
            x = np.linspace(cx - 0.5*span_x, cx + 0.5*span_x, n_sensors)
            y = np.full(n_sensors, cy)

        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        r_posicion = np.zeros((3, n_sensors))
        r_posicion[0, :] = x
        r_posicion[1, :] = y
        r_posicion[2, :] = hr0
        return r_posicion

    def generate_trajectories(self):
        """Generate trajectories or use precomputed ones.

        Dispatches to a method-specific submethod selected by
        ``self.traj_config.method`` so new trajectory shapes can be added
        without touching the channel-processing code. To add a method, add a
        parameter block to config/data_pipeline.yaml and a matching
        ``_generate_<method>_trajectories`` submethod here.
        """
        if self.precomputed_trajectories is not None:
            return self.precomputed_trajectories

        assert self.params is not None, "params must be initialized"
        method = self.traj_config['method'] #type: ignore
        if method == 'spiral':
            return self._generate_spiral_trajectories()
        elif method == 'hermite':
            return self._generate_hermite_trajectories()
        else:
            raise ValueError(f"Unknown trajectory generation method: {method}")

    def _generate_spiral_trajectories(self):
        """Original outward-spiral trajectories.

        Parameters are read from the ``spiral`` block of the trajectory
        config (config/data_pipeline.yaml).
        """
        assert self.params is not None, "params must be initialized"
        n_traj = self.params['n_traj']
        ppt = self.params['ppt']
        cfg = self.traj_config['spiral'] #type: ignore

        radio_t = np.random.uniform(cfg['radio_min'], cfg['radio_max'], size=n_traj)
        fase0 = 2 * np.pi * np.random.rand(n_traj)
        omega0 = np.random.uniform(cfg['omega_min'], cfg['omega_max'], size=n_traj)

        traj = np.zeros([3, n_traj, ppt + 1])
        aux_1 = np.linspace(0, cfg['angle_span'] * ppt, ppt + 1)
        aux_2 = np.linspace(cfg['radius_start'] * ppt, cfg['radius_end'] * ppt, ppt + 1)

        for it in range(n_traj):
            traj[0, it, :] = radio_t[it] / ppt * aux_2 * np.cos(
                omega0[it] / ppt * aux_1 + fase0[it])
            traj[1, it, :] = radio_t[it] / ppt * aux_2 * np.sin(
                omega0[it] / ppt * aux_1 + fase0[it])
            traj[2, it, :] = 0

        return traj

    def _generate_hermite_trajectories(self):
        """Free-tangent piecewise-cubic Hermite trajectories.

        Each trajectory is built from ``n_segments`` cubic Hermite segments
        joined with C1 continuity (matched position *and* tangent at every
        knot, so the path is smooth in xy). Parameters are read from the
        ``hermite`` block of config/data_pipeline.yaml.

        Design notes
        ------------
        * The knot tangent *directions* form a slowly-turning sequence: the
          first is uniform over all directions, and each subsequent one differs
          from the previous by at most ``max_turn`` radians. This keeps the
          cosine distance between consecutive tangents small -- no sudden
          turn-arounds -- while still allowing the path to wander.
        * Waypoints step along those same directions, so each segment leaves a
          knot in its tangent direction (no cusps / overshoot).
        * Tangent magnitudes are tied to the adjacent segment lengths
          (Catmull-Rom-style tension scaling), so segment length controls speed
          and the curve stays well-shaped. Speed therefore varies along the
          trajectory, as it already does for the spiral.

        Why this suits the context task: the first few points lie on the first
        segment and reveal roughly the first knot/tangent, but the later
        waypoints are sampled independently and stay hidden -- so the first 5
        context points do not determine the rest of the trajectory.
        """
        assert self.params is not None, "params must be initialized"
        n_traj = self.params['n_traj']
        ppt = self.params['ppt']
        cfg = self.traj_config['hermite'] #type: ignore

        K = int(cfg['n_segments'])          # number of segments
        n_knots = K + 1                     # waypoints / knots
        max_turn = float(cfg['max_turn'])
        len_min = float(cfg['seg_len_min'])
        len_max = float(cfg['seg_len_max'])
        tension = float(cfg['tension'])
        start_r_min = float(cfg['start_radius_min'])
        start_r_max = float(cfg['start_radius_max'])

        def hermite_segment(P0, M0, P1, M1, u):
            """Evaluate a cubic Hermite segment at local params u in [0, 1]."""
            u2 = u * u
            u3 = u2 * u
            h00 = 2 * u3 - 3 * u2 + 1
            h10 = u3 - 2 * u2 + u
            h01 = -2 * u3 + 3 * u2
            h11 = u3 - u2
            return (h00[:, None] * P0 + h10[:, None] * M0
                    + h01[:, None] * P1 + h11[:, None] * M1)

        traj = np.zeros([3, n_traj, ppt + 1])

        # Knot times 0, 1, ..., K and the global sampling parameter in [0, K].
        t_samples = np.linspace(0.0, K, ppt + 1)
        seg_idx = np.clip(np.floor(t_samples).astype(int), 0, K - 1)
        u_local = t_samples - seg_idx       # local param within each segment

        for it in range(n_traj):
            # --- slowly-turning knot tangent directions ---
            thetas = np.empty(n_knots)
            thetas[0] = 2 * np.pi * np.random.rand()            # first: any direction
            thetas[1:] = np.random.uniform(-max_turn, max_turn, size=K)
            thetas = np.cumsum(thetas)                          # bounded random walk
            dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)  # (n_knots, 2)

            # --- waypoints: step along the knot directions ---
            # Start point sampled like the spiral's first point: random angle at
            # a radius in [start_radius_min, start_radius_max] (not the origin).
            start_r = np.random.uniform(start_r_min, start_r_max)
            start_a = 2 * np.pi * np.random.rand()
            seg_len = np.random.uniform(len_min, len_max, size=K)
            P = np.zeros((n_knots, 2))
            P[0] = [start_r * np.cos(start_a), start_r * np.sin(start_a)]
            for k in range(K):
                P[k + 1] = P[k] + seg_len[k] * dirs[k]

            # --- knot tangent magnitudes (tension-scaled by adjacent lengths) ---
            mag = np.empty(n_knots)
            mag[0] = seg_len[0]
            mag[-1] = seg_len[-1]
            mag[1:-1] = 0.5 * (seg_len[:-1] + seg_len[1:])
            M = tension * mag[:, None] * dirs                   # (n_knots, 2)

            # --- evaluate the piecewise-Hermite curve ---
            xy = hermite_segment(P[seg_idx], M[seg_idx],
                                 P[seg_idx + 1], M[seg_idx + 1], u_local)
            traj[0, it, :] = xy[:, 0]
            traj[1, it, :] = xy[:, 1]
            traj[2, it, :] = 0

        return traj

    def obtain_h(self):
        # Generar (o cargar) trayectorias
        assert self.params is not None, "params must be initialized"
        self.traj = self.generate_trajectories()
    
        # Generar sensores adaptados al tamaño real de las trayectorias
        self.r_posicion = self.generate_sensor_positions(self.traj, scale=0.6, min_span=20.0)
           
        # Process values for each pair (sensor, trajectory)
        def process_sensor(ise):
            assert self.params is not None, "params must be initialized"
            import scipy.signal as signal
            Lf = len(range_m(self.params['ci']['fmin'], self.params['ci']['fmax'], self.params['ci']['df']))
            h_val = np.zeros([self.params['n_traj'], self.params['ppt'], Lf],
                             dtype=np.complex128)
            
            for itx in tqdm(range(self.params['n_traj']), 
                          desc=f"Sensor {ise+1} processing trajectories ({self.topology})", 
                          leave=False):
                assert self.traj is not None, "trajectories (traj) must be initialized"
                assert self.r_posicion is not None, "sensor positions (r_posicion) must be initialized"
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
                            else:
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
        assert self.params is not None, "params must be initialized"
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
            assert self.params is not None, "params must be initialized"
            assert self.h is not None, "channel matrix (h) must be initialized"
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

        assert self.h is not None, "channel matrix (h) must be initialized"
        assert self.traj is not None, "trajectories (traj) must be initialized"
        out = list(out)  # Convert generator to list if needed
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
                    'df': 50, #25.000000,  # frequency_resolution_[Hz]
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
            aux = 1.0  # Default for non-'no_var' strings
    else:
        aux = options if options is not None else 1.0
    
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
              'ppt': 30,#50
              'm': 0.7,
              'T': 20,
              }
    params['w0'] = np.pi / params['T']
    return params

def plot_validation(channels_dict, option_idx=0, n_samples=10, seed=None):
    """
    Genera n_samples imágenes de comparación de topologías
    para trayectorias elegidas aleatoriamente.
    Cada imagen muestra la MISMA trayectoria en las 3 topologías.
    Se muestra el numero de cada sensor y se marcan claramente el punto de inicio (verde) y el punto final (negro) de la trayectoria.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    topologies = ['ellipsoidal', 'random', 'aligned']
    titles = ['Ellipsoidal Topology', 'Random Topology', 'Aligned Topology']

    # cuántas trayectorias hay en total (lo tomamos de la primera topología)
    n_traj = channels_dict[topologies[0]].traj.shape[1]

    # generador aleatorio
    rng = np.random.default_rng(seed)
    # elegimos hasta n_samples trayectorias distintas
    chosen_trajs = rng.choice(n_traj, size=min(n_samples, n_traj), replace=False)

    # carpeta base donde ya guardabas
    base_dir = 'data'
    plot_dir = f'{base_dir}/validation'
    # NUEVA subcarpeta
    samples_dir = os.path.join(plot_dir, f'random_trajectories_option_{option_idx}')
    os.makedirs(samples_dir, exist_ok=True)

    # Para que todas las imágenes tengan el mismo encuadre,
    # primero calculamos los límites globales UNA VEZ
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    for topology in topologies:
        c = channels_dict[topology]
        traj_all = c.traj  # shape: (2, n_traj, T) o similar
        for traj_idx in chosen_trajs:
            traj = traj_all[:, traj_idx, :]
            x_min = min(x_min, traj[0, :].min())
            x_max = max(x_max, traj[0, :].max())
            y_min = min(y_min, traj[1, :].min())
            y_max = max(y_max, traj[1, :].max())
            # también miramos sensores
            sensors = c.r_posicion
            x_min = min(x_min, sensors[0, :].min())
            x_max = max(x_max, sensors[0, :].max())
            y_min = min(y_min, sensors[1, :].min())
            y_max = max(y_max, sensors[1, :].max())

    # ahora sí: generamos una figura por trayectoria elegida
    for k, trajectory_idx in enumerate(chosen_trajs):
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        for idx, (ax, topology, title) in enumerate(zip(axes, topologies, titles)):
            c = channels_dict[topology]
            traj = c.traj[:, trajectory_idx, :]
            sensors = c.r_posicion

            # trayectoria
            ax.plot(traj[0, :], traj[1, :], 'b-', linewidth=3)
            # punto inicio
            ax.plot(traj[0, 0], traj[1, 0], 'go', markersize=17)
            # punto final
            ax.plot(traj[0, -1], traj[1, -1], 'ko', markersize=17)
            # sensores junto con su número
            for i, (x, y) in enumerate(zip(sensors[0, :], sensors[1, :])):
                ax.text(x, y, str(i), fontsize=15, ha='center', va='center', color='white', weight='bold')
                ax.plot(x, y, 'ro', markeredgecolor='black', markersize=17)

            #ax.set_title(f"{title}\n(traj {trajectory_idx})", fontsize=20)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.4)
            ax.tick_params(axis='both', labelsize=18)

            if idx == 0:
                ax.plot([], [], 'b-', linewidth=3, label='Trajectory')
                ax.plot([], [], 'go', markersize=17, label='Start')
                ax.plot([], [], 'ko', markersize=17, label='End')
                ax.plot([], [], 'ro', markeredgecolor='black', markersize=17, label='Sensors')
                ax.legend(loc='upper left', fontsize=18)

        plt.tight_layout()

        # nombre de la imagen
        out_path = os.path.join(
            samples_dir,
            f'topology_comparison_option_{option_idx}_traj_{int(trajectory_idx):03d}.png'
        )
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Se han guardado {len(chosen_trajs)} imágenes en: {samples_dir}")

def save_velocity_histogram(traj, T_tot, bins=40, option_idx=None):
    """
    Calcula el histograma de velocidades a partir de las trayectorias generadas
    y guarda la figura en el directorio actual.

    Parámetros
    ----------
    traj : np.ndarray
        Array de trayectorias con forma (3, n_traj, ppt+1).
    T_tot : float
        Tiempo T_tot usado actualmente en el canal para la velocidad "vehicular".
        IMPORTANTE: aquí se interpreta como tiempo por salto (tal y como está en tu código).
    bins : int
        Número de bins del histograma.
    filename : str
        Nombre del archivo de imagen a guardar (PNG).

    Retorna
    -------
    dict con estadísticas básicas: {'mean': ..., 'min': ..., 'max': ..., 'std': ...}
    """
    assert traj.ndim == 3 and traj.shape[0] == 3, "traj debe ser de forma (3, n_traj, ppt+1)"
    # Diferencias entre puntos consecutivos (componentes x,y,z)
    diffs = np.diff(traj, axis=2)                 # -> (3, n_traj, ppt)
    # Módulo del desplazamiento por salto
    step_dists = np.linalg.norm(diffs, axis=0)    # -> (n_traj, ppt)
    # Velocidades por salto con la semántica actual: v = Δs / T_tot
    speeds = step_dists / float(T_tot)            # -> (n_traj, ppt)
    speeds_flat = speeds.ravel()

    # Estadísticas
    v_mean = float(np.mean(speeds_flat))
    v_min  = float(np.min(speeds_flat))
    v_max  = float(np.max(speeds_flat))
    v_std  = float(np.std(speeds_flat))

    # Figura
    plt.figure(figsize=(8, 5))
    plt.hist(speeds_flat, bins=bins, edgecolor="k")
    plt.xlabel("Velocidad por salto (m/s)")
    plt.ylabel("Frecuencia")
    plt.title("Histograma de velocidades (Δs / T_tot)")
    # Texto con stats
    txt = f"mean={v_mean:.2f} m/s\nstd={v_std:.2f} m/s\nmin={v_min:.2f} m/s\nmax={v_max:.2f} m/s"
    plt.gca().text(0.98, 0.95, txt, ha="right", va="top", transform=plt.gca().transAxes,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5))
    plt.tight_layout()

    # Guardado en el directorio de ejecución
    base_dir = 'data'
    plot_dir = f'{base_dir}/validation'
    filename=f"velocity_hist_theta_{option_idx}.png"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

    #print(f"Histograma guardado en: {base_dir}/validation/{filename}")
    #print(f"Velocidad media: {v_mean:.3f} m/s | std: {v_std:.3f} m/s | min: {v_min:.3f} m/s | max: {v_max:.3f} m/s")

    return {"mean": v_mean, "std": v_std, "min": v_min, "max": v_max}

def process(channel_options, snr, rep, nop=-1, n_traj_override=None,
            ppt_override=None, df_override=None, topologies=None,
            out_dir='data', method=None, variant_subdir=True,
            master_seed=11, traj_config=None):
    """Process data for all topologies using the same trajectories.

    All requested topologies share ONE trajectory ensemble per theta; the
    trajectory shape (spiral / hermite) is selected by ``method`` (falling back
    to ``traj_config['method']``). The output layout, per (topology, theta), is::

        <out_dir>/<topology>/<method>/channel_option_<theta>/
            trajectory/trajectories.npy          (3, n_traj, ppt)   target coords
            filtered_data/filtered_data.npy       (tau, ppt, n_traj, n_sensors)
            channel_info/sensor_positions_<theta>.npy   (3, n_sensors)
            channel_info/trajs_<theta>.npy        (3, n_traj, ppt+1) full traj

    With ``variant_subdir=False`` the ``<method>`` level is omitted (legacy
    flat layout). The ``channel_info`` copies let the QC / velocity scripts
    verify the sensor layouts and recover per-jump velocity (needs the ppt+1
    endpoint) without re-running the physics.
    """
    if topologies is None:
        topologies = ['ellipsoidal', 'random', 'aligned']

    # Trajectory-generation config (loaded from config/data_pipeline.yaml).
    if traj_config is None:
        traj_config = load_traj_config()
    if method is None:
        method = traj_config.get('method', 'spiral') #type: ignore

    for option_idx, option in enumerate(tqdm(channel_options)):
        np.random.seed(master_seed)
        print(f"\n\n --- Processing Channel Option: {option} ---")

        # Generate parameters for this option
        params = generate_params(options=option)

        if n_traj_override is not None:
            params['n_traj'] = int(n_traj_override)
        if ppt_override is not None:
            params['ppt'] = int(ppt_override)
        if df_override is not None:
            params['ci']['df'] = float(df_override)
        # First, generate trajectories that will be shared across all topologies
        # We create a temporary channel just to generate trajectories
        temp_channel = channel(load=False, params=params, number_of_processes=nop,
                              name=str(option), topology='ellipsoidal',
                              traj_config=traj_config)
        shared_trajectories = temp_channel.traj  # Store the trajectories
        
        channels_for_validation = {}
        
        # Initialize trjs to avoid unbound variable error
        trjs = None
        data = None
        
        # Process each topology with the same trajectories
        for topology in topologies:
            print(f"\nProcessing topology: {topology}")
            
            # Create channel with specific topology and shared trajectories
            c = channel(load=False, params=params, number_of_processes=nop,
                       name=str(option), topology=topology,
                       precomputed_trajectories=shared_trajectories,
                       traj_config=traj_config)
            
            # Store for validation plotting
            channels_for_validation[topology] = c
            
            # Generate the filtered features + target coordinates. We pass an
            # explicit canonical ordering via `specific=` so the trajectory rows
            # (and their matched features) are stored in the SAME order for every
            # topology. Without it, channel.filter() draws a RANDOM trajectory
            # ordering (np.random.choice over n_traj) per call, which would store
            # each topology's trajectories in a different row order -- breaking
            # row-alignment across geometries (trajectory i in 'ellipsoidal' would
            # not correspond to trajectory i in 'random') even though they share
            # the same underlying ensemble.
            n_traj_c = c.params['n_traj']
            canonical = list(range(n_traj_c))
            data, trjs = c.filter(1024, snr=snr, nt=n_traj_c, signal_type='sinusoid',
                                  rep=rep, specific=canonical)
            
            # Save the generated data. New layout puts the topology FIRST, then
            # the method, then the channel option:
            #   <out_dir>/<topology>/<method>/channel_option_<theta>/...
            if variant_subdir:
                topology_dir = f'{out_dir}/{topology}/{method}/channel_option_{option}'
            else:
                topology_dir = f'{out_dir}/{topology}/channel_option_{option}'
            info_dir = f'{topology_dir}/channel_info'

            # Create directories for this topology
            os.makedirs(f'{topology_dir}/trajectory', exist_ok=True)
            os.makedirs(f'{topology_dir}/filtered_data', exist_ok=True)
            os.makedirs(info_dir, exist_ok=True)

            # Save data
            assert data is not None, "Filtered data is None"
            assert trjs is not None, "Trajectories are None"
            np.save(f'{topology_dir}/trajectory/trajectories.npy', trjs)
            np.save(f'{topology_dir}/filtered_data/filtered_data.npy', data)
            # Sensor layout + full (ppt+1) trajectory for the QC / velocity tools.
            np.save(f'{info_dir}/sensor_positions_{option}.npy', c.r_posicion)
            np.save(f'{info_dir}/trajs_{option}.npy', c.traj)

            print(f" Data saved for topology: {topology}")
        
        # Create validation plot for this option (only for the first option)
        if option_idx == 0:
            plot_validation(channels_for_validation, option)
        
        assert trjs is not None, "Trajectories must be set after topology loop"
        stats = save_velocity_histogram(trjs, T_tot=params['ci']['T_tot'], bins=40, option_idx=option)

def parse_float_list(s):
    if s is None:
        return None
    parts = [p for p in s.replace(',', ' ').split() if p != '']
    return [float(p) for p in parts]

def run_topology_task(cfg):
    """Generate the TOPOLOGY-task datasets from a unified config object
    (config/data_pipeline.yaml). Reads the shared ``channel`` block, the
    ``topology_task`` block and ``method``; writes one dataset per topology
    (all sharing the same trajectories + channels) under
    ``topology_task.out_dir``. Invoked by data/generate.py (task=topology)."""
    ch = cfg['channel']
    tt = cfg['topology_task']
    method = cfg.get('method', 'spiral')

    channel_options = [float(x) for x in ch['channel_options']]
    topologies = [str(t) for t in tt['topologies']]
    out_dir = str(tt['out_dir'])
    n_traj = int(ch['n_traj']); ppt = int(ch['ppt']); df = float(ch['df'])
    snr = float(ch['snr']); rep = int(ch['rep'])
    master_seed = int(ch.get('master_seed', 11)); nop = int(ch.get('nop', -1))

    # Suppress channel.__init__'s hardcoded ./data side-effect write; the real
    # channel_info is saved by process() into the proper out_dir.
    channel.save_channel_info = lambda self, name: None  # type: ignore

    Lf = len(range_m(10000.0, 20000.0, df)); n_sensors = 10
    print("=" * 64)
    print("TOPOLOGY task -- dataset generation")
    print("=" * 64)
    print(f"  thetas        : {channel_options}")
    print(f"  topologies    : {topologies}")
    print(f"  n_traj / ppt  : {n_traj} / {ppt}")
    print(f"  df            : {df} Hz  ->  Lf={Lf} time-points")
    print(f"  feature/point : Lf*n_sensors = {Lf}*{n_sensors} = {Lf*n_sensors}")
    print(f"  snr / rep     : {snr} / {rep}")
    print(f"  traj method   : {method}")
    print(f"  master seed   : {master_seed}")
    print(f"  layout        : {out_dir}/<topology>/{method}/channel_option_<theta>/")
    print("=" * 64)

    np.random.seed(master_seed)
    process(channel_options, snr=snr, rep=rep, nop=nop,
            n_traj_override=n_traj, ppt_override=ppt, df_override=df,
            topologies=topologies, out_dir=out_dir, method=method,
            variant_subdir=True, master_seed=master_seed, traj_config=cfg)


if __name__ == '__main__':
    print("acoustic_data_generator.py is now a library used by the unified "
          "generator. Run:\n  python data/generate.py task=topology")