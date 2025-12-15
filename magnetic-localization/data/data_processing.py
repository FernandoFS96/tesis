import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def load_data(file_path):
    """
    Carga el archivo de datos usando pandas, asumiendo que los datos están separados por espacios.
    """
    try:
        df = pd.read_table(file_path, sep="\s+", header=None)
        print("Datos cargados desde:", file_path)
        return df
    except Exception as e:
        print("Error al cargar datos:", e)
        raise

def obtain_b_matrix(values):
    """
    Procesa los datos para construir la matriz B:
      - Extrae los valores únicos de X (columna 0) e Y (columna 1).
      - Asigna el valor de la columna 3 en la posición correspondiente.
      - Aplica una transformación logarítmica preservando el signo.
    """
    x_vals = np.unique(values[:, 0])
    y_vals = np.unique(values[:, 1])
    nx = len(x_vals)
    ny = len(y_vals)
    
    B_matrix = np.zeros((nx, ny))
    for row in values:
        x_val = row[0]
        y_val = row[1]
        b_value = row[3]  # Se asume que la columna 3 tiene el valor del campo magnético
        i = np.where(x_vals == x_val)[0][0]
        j = np.where(y_vals == y_val)[0][0]
        B_matrix[i, j] = b_value
    
    epsilon = 1e-14  # Para evitar log(0)
    B_transformed = -np.sign(B_matrix) * np.log(np.abs(B_matrix) + epsilon)
    
    return B_transformed, x_vals, y_vals

def generate_curved_trajectory(num_points=100, amplitude=20, frequency=0.05, phase=0.0, offset=(0,0)):
    """
    Genera una trayectoria curva.
    En este ejemplo, se utiliza una función seno para la componente y y un espaciamiento lineal para x.
    """
    x = np.linspace(-90, 90, num_points)
    y = offset[1] + amplitude * np.sin(frequency * x + phase)
    return np.vstack((x, y)).T

def fixed_sensor_positions(center=(0,0), radius=50, num_sensores=4):
    """
    Define las posiciones fijas de los sensores en el entorno.
    En este ejemplo, se colocan en un arreglo circular alrededor de 'center' con un radio dado.
    """
    angulos = np.linspace(0, 2*np.pi, num_sensores, endpoint=False)
    positions = []
    for theta in angulos:
        sensor_x = center[0] + radius * np.cos(theta)
        sensor_y = center[1] + radius * np.sin(theta)
        positions.append((sensor_x, sensor_y))
    return np.array(positions)

def create_dataset_fixed_sensors(trajectory, interpolator, sensor_positions_fixed):
    """
    Para cada instante de la trayectoria, se simula la lectura de cada sensor fijo.
    Se asume que el objeto genera una firma magnética que se desplaza con él.
    Por ello, la lectura del sensor en posición fija S es:
       lectura = interpolator(S - objeto)
    Cada muestra se guarda con la forma:
       [object_x, object_y, sensor_1, sensor_2, ..., sensor_n]
    """
    data = []
    for pos_obj in trajectory:
        readings = []
        for sensor_pos in sensor_positions_fixed:
            # Posición relativa del sensor respecto al objeto
            relative_pos = sensor_pos - pos_obj
            reading = interpolator(relative_pos).item()
            readings.append(reading)
        sample = np.concatenate([pos_obj, readings])
        data.append(sample)
    return np.array(data)

def plot_trajectory_and_sensors(trajectory, sensor_positions_fixed, save_path=None):
    """
    Dibuja la trayectoria del objeto junto con la posición fija de los sensores
    y etiqueta cada sensor con su número correspondiente.
    """
    plt.figure(figsize=(8,6))
    # Plotea la trayectoria
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Trayectoria')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Inicio')
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='Fin')
    
    # Plotea los sensores fijos
    plt.plot(sensor_positions_fixed[:, 0], sensor_positions_fixed[:, 1], 'ks', markersize=10, label='Sensores fijos')
    
    # Añade etiquetas a cada sensor
    for idx, (x, y) in enumerate(sensor_positions_fixed):
        plt.text(x, y, f'{idx+1}', fontsize=12, color='k', ha='right', va='bottom')
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trayectoria y sensores fijos")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
        print("Gráfico guardado en:", save_path)
    plt.close()

def process_file(file_path, output_base_dir, n_trayectorias=50, n_sensores=4):
    """
    Procesa un archivo .dat: genera n_trayectorias curvas, crea el dataset con lecturas de sensores fijos,
    y guarda los resultados (CSV y 10 plots) en una subcarpeta cuyo nombre se basa en el archivo.
    """
    df = load_data(file_path)
    data_array = df.values
    B_transformed, x_vals, y_vals = obtain_b_matrix(data_array)
    
    # Crear interpolador para el campo magnético transformado
    interpolator = RegularGridInterpolator((x_vals, y_vals), B_transformed, bounds_error=False, fill_value=None)
    
    # Crear carpeta de salida para este archivo, usando su nombre (sin extensión)
    filename = os.path.basename(file_path)
    filename_no_ext = os.path.splitext(filename)[0]
    output_dir = os.path.join(output_base_dir, filename_no_ext)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Carpeta creada:", output_dir)
    
    # Posiciones fijas de los sensores (configurables)
    sensores_fijos = fixed_sensor_positions(center=(0,0), radius=50, num_sensores=n_sensores)
    
    all_data = []           # Para almacenar los datos de todas las trayectorias
    trajectories_list = []  # Para guardar cada trayectoria (para los plots)
    
    # Generar n_trayectorias curvas con parámetros aleatorios
    for i in range(n_trayectorias):
        amplitude = np.random.uniform(15, 30)
        frequency = np.random.uniform(0.03, 0.07)
        phase = np.random.uniform(0, 2*np.pi)
        offset_y = np.random.uniform(-10, 10)
        traj = generate_curved_trajectory(num_points=100, amplitude=amplitude,
                                          frequency=frequency, phase=phase,
                                          offset=(0, offset_y))
        trajectories_list.append(traj)
        dataset = create_dataset_fixed_sensors(traj, interpolator, sensores_fijos)
        traj_ids = np.full((dataset.shape[0], 1), i)
        dataset_with_id = np.hstack((traj_ids, dataset))
        all_data.append(dataset_with_id)
    
    all_data = np.vstack(all_data)
    columnas = ['traj_id', 'object_x', 'object_y'] + [f'sensor_{j+1}' for j in range(n_sensores)]
    df_all = pd.DataFrame(all_data, columns=columnas)
    csv_path = os.path.join(output_dir, f"dataset.csv")
    df_all.to_csv(csv_path, index=False)
    print("Dataset guardado en:", csv_path)

    # Graficar 10 trayectorias y sensores
    for i, traj in enumerate(trajectories_list[:10]):
        plot_path = os.path.join(output_dir, f"trajectory_{i+1}.png")
        plot_trajectory_and_sensors(traj, sensores_fijos, save_path=plot_path)

def main():
    # Directorio relativo donde se encuentran los archivos .dat (datos crudos)
    input_dir = os.path.join(os.getcwd(), "simulated_data")
    
    # Directorio base de salida relativo para guardar los resultados
    output_base_dir = os.path.join(os.getcwd(), "trajectories")
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print("Carpeta base de salida creada:", output_base_dir)
        
    # Obtener todos los archivos .dat en el directorio de entrada
    dat_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".dat")]
    print("Archivos .dat encontrados:", dat_files)
    
    # Procesar cada archivo .dat
    for file_path in dat_files:
        process_file(file_path, output_base_dir, n_trayectorias=100, n_sensores=6)

if __name__ == "__main__":
    main()