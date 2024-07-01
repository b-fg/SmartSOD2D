#!/usr/bin/env python3
import numpy as np

def n_witness_points(fname):
    return sum([1 for l in open(fname,"r").readlines() if l.strip()])


def n_rectangles(fname):
    return int(open(fname,"r").readline())


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_params(params, title=None):
    if title: print(f"{bcolors.OKGREEN}{title}{bcolors.ENDC}")
    print(params_str(params))


def params_str(params, title=None):
    my_str = ""
    if title:
        my_str += "title\n"
    for k,v in params.items():
        my_str += f"\n{k}: {v}"
    my_str += "\n"
    return my_str


def numpy_str(a, precision=2):
    return np.array2string(a, precision=precision, floatmode='fixed')


def generate_witness_points_coordinates(filename='witness.txt', xyz_size=(6, 6, 6), x_lims=(180, 420), y_lims=(5, 55), z_domain=125, noise_amplitude=0.01):
    def cartesian_product(arrays):
        la = len(arrays)
        dtype = np.find_common_type([a.dtype for a in arrays], [])
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    z_lims = (z_domain / xyz_size[2] - z_domain / (2 * xyz_size[2]), z_domain - z_domain / (2 * xyz_size[2]))
    x_vec = np.linspace(x_lims[0], x_lims[1], xyz_size[0])
    y_vec = np.linspace(y_lims[0], y_lims[1], xyz_size[1])
    z_vec = np.linspace(z_lims[0], z_lims[1], xyz_size[2])

    coords = cartesian_product([z_vec, y_vec, x_vec]) # x will vary fastest, but will be on 3rd columns
    coords[:, [0, 2]] = coords[:, [2, 0]] # swap x and z columns (restore conventional order)
    coords += np.random.rand(*coords.shape) * noise_amplitude

    np.savetxt(filename, coords, '%.4f')
    print(f"Cube coordinates have been exported to {filename} successfully.")

    return coords

def read_witness_points_coordinates(filename='witness.txt'):
    return np.loadtxt(filename, delimiter=' ', dtype=np.float32)


def witness_points_normalization(filename_in='resultwit.h5', filename_out='witness_norm.txt', fields=('u_x', 'u_y', 'u_z', 'pr')):
    """
    Postprocesses a witness point h5 output file of a non-actuated case and computes mean and std for each point.
    Then writes this information into a txt file while is finally used in an actuated case to normalise the state.
    Statistics computed in velocity and pressure field. Output looks like: (field, n_wit (mean, std))
    """
    h5file = h5py.File(filename_in)
    data_shape = np.array(h5file.get('u_x')).shape
    n_wit = data_shape[0]
    mean_std_data = np.empty((len(fields), n_wit, 2))
    for i,var in enumerate(fields):
        data = np.array(h5file.get(var))
        mean_std_data[i, :, 0] = np.mean(data, axis=-1)
        mean_std_data[i, :, 1] = np.std(data, axis=-1)

    mean_std_data = np.reshape(mean_std_data, (len(fields), n_wit * 2))
    np.savetxt(filename_out, mean_std_data, '%7.4e')
    print(f"Witness points normalization data have been exported to {filename_out} successfully.")


# TensorFlow utils
def deactivate_tf_gpus():
    import tensorflow as tf

    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'