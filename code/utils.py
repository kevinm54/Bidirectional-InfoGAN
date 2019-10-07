import numpy as np
import tensorflow as tf
from tensorflow.python.layers import utils

def log(x):
    return tf.log(x + 1e-8)


def get_max_idx(max, a):
    return [i for i, j in enumerate(a) if j == max][0]


def get_min_idx(min, a):
    return [i for i, j in enumerate(a) if j == min][0]


def sample_z(num_samples, n):
    return np.random.uniform(-1., 1., size=[num_samples, n])


# weight initialization functions
def he_init(size, dtype=tf.float32, partition_info=None):
    in_dim = size[0]
    he_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=he_stddev)

truncated_normal = tf.truncated_normal_initializer(stddev=0.02)


def gaussian_noise_layer(input_layer, std, training):
    def add_noise():
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    return utils.smart_cond(training, add_noise, lambda: input_layer)


def sample_z_fixed(num_samples, n):
    z_out = np.zeros((num_samples, n))
    for idx in range(10):
        z_out[idx * 10:idx * 10 + 10, :] = np.random.uniform(-1., 1., size=[1, n])
    return z_out


def sample_c(num_samples, num_cont_vars=2, disc_classes=[10]):
    """
    Sample a random c vector, i.e. categorical and continuous variables.
    If test is True, samples a value for each discrete variable and combines each with the chosen continuous
    variable c; all other continuous c variables are sampled once and then kept fixed
    """
    c = np.random.multinomial(1, disc_classes[0] * [1.0 / disc_classes[0]], size=num_samples)
    for cla in disc_classes[1:]:
        c = np.concatenate((c, np.random.multinomial(1, cla * [1.0 / cla], size=num_samples)), axis=1)
    for n in range(num_cont_vars):
        cont = np.random.uniform(-1, 1, size=(num_samples, 1))
        c = np.concatenate((c, cont), axis=1)
    return c


def sample_c_cat(num_samples=128, disc_var=0, num_cont_vars=2, num_disc_vars=10, disc_classes=[10]):
    """
    Samples categorical values for visualization purposes
    """
    # cont = []
    # cont_matr = []
    # for idx in range(num_cont_vars):
    #     cont.append(np.random.uniform(-1, 1, size=[1, 10]))
    #     cont_matr.append(np.zeros(shape=(num_samples)))
    
    # print("num_samples = ", num_samples)
    # print("shape of cont: ", np.shape(cont))
    # print("shape of cont_matr: ", np.shape(cont_matr))

    # for idx in range(num_cont_vars):
    #     for idx2 in range(10):
    #         print("idx = %d, idx2 = %d:" % (idx, idx2))
    #         print("  shape(cont[idx][0,idx2]): ", np.shape(cont[idx][0,idx2]))
    #         print("  shape(cont_matr[idx][10 * idx2: 10 * idx2 + 10]): ",
    #             np.shape(cont_matr[idx][10 * idx2: 10 * idx2 + 10]))
    #         cont_matr[idx][10 * idx2: 10 * idx2 + 10] = np.broadcast_to(cont[idx][0, idx2], (10))
    
    cont_matr = np.random.uniform(-1, 1, size=[num_cont_vars, num_samples])

    cs_cont = np.zeros(shape=(num_samples, num_cont_vars))

    for idx in range(num_cont_vars):
        cs_cont[:, idx] = cont_matr[idx]

    c = np.eye(10, 10)
    for idx in range(1, 10):
        c_tmp = np.eye(10, 10)
        c = np.concatenate((c, c_tmp), axis=0)

    counter = 0
    cs_disc = np.zeros(shape=(num_samples, num_disc_vars))
    for idx, cla in enumerate(disc_classes):
        if idx == disc_var:
            tmp = np.zeros((num_samples, cla))
            for sample in range(num_samples):
                tmp[sample,sample%cla] = 1
            cs_disc[:, counter:counter + cla] = tmp
            counter += cla
        else:
            tmp = np.zeros(shape=(num_samples, cla))
            for sample in range(num_samples):
                cat_id = np.random.randint(0, cla)
                tmp[sample, cat_id] = 1
            cs_disc[:, counter:counter + cla] = tmp
            counter += cla

    c = np.concatenate((cs_disc, cs_cont), axis=1)

    return c


def sample_c_cont(sample_count=128, num_cont_vars=2, num_disc_vars=10, c_dim=12, c_var=0, c_const=[1]):
    """
    Samples continuous values for visualization purposes
    """
    assert(c_dim is num_cont_vars+num_disc_vars)
    # cont = [-2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0]
    cont = [-1.0, -0.8, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 0.8, 1.0]
    num_cont_samples = len(cont)

    c_cont = np.zeros((num_cont_samples, num_cont_vars))
    for idx in range(num_cont_vars):
        if idx == c_var:
            c_cont[:, c_var] = cont
        else:
            c_cont[:, idx] = np.random.uniform(-1,1, size=num_cont_samples)

    c_out = np.zeros((sample_count, c_dim))
    for idx in range(sample_count):
        disc_class = np.random.randint(0,num_disc_vars)
        c_out[idx, disc_class] = 1
        c_out[idx, num_disc_vars:] = c_cont[idx % num_cont_samples, :]

    return c_out
