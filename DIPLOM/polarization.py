import numpy as np


def raschet(uni, normal, tangent, bitangent):
    cross_mul = np.cross(uni, normal)
    null = np.array([0, 0, 0])
    if np.array_equal(cross_mul, null):
        s = tangent
        p = bitangent
    else:
        s = get_s(uni, normal)
        p = get_p(uni, s)

    return s, p


def new_tangent_bitangent(uni, s):
    length = (np.cross(uni, s).dot(np.cross(uni, s))) ** 0.5
    p = np.cross(uni, s) / length
    return s, p


def Es_Ep(jv, s, p, e):
    old_jv = np.array([0, jv[0], jv[1]])
    mat = np.array([[e[0], e[1], e[2]],
                   [s[0], s[1], s[2]],
                   [p[0], p[1], p[2]]])
    inv_mat = np.linalg.inv(mat)
    new_jv = np.dot(inv_mat, old_jv)

    return new_jv


def E_s(E_t, tangent, s, E_b, bitangent):
    return E_t * np.dot(tangent, s) + E_b * np.dot(bitangent, s)


def E_p(E_t, tangent, E_b, bitangent, p):
    return E_t * np.dot(tangent, p) + E_b * np.dot(bitangent, p)


def basis_of_ray(dir_vec):
    # tangent = np.array([-dir_vec[2], 0, dir_vec[0]])
    # cross = np.cross(dir_vec, tangent)
    # length = (np.dot(cross, cross)) ** 0.5
    # bitangent = np.cross(dir_vec, tangent) / length
    tangent = np.array([0, 1, 0])
    bitangent = np.array([0, 0, 1])
    return tangent, bitangent


def get_s(uni, normal):
    length = (np.cross(uni, normal).dot(np.cross(uni, normal))) ** 0.5
    return np.cross(uni, normal) / length


def get_p(uni, s):
    length = (np.cross(uni, s).dot(np.cross(uni, s))) ** 0.5
    return np.cross(uni, s) / length


# def E_reflect_s(n1, n2, teta_i):
#     cos_teta_i = np.cos(teta_i)
#     teta_t = np.arcsin(n1 * np.sin(teta_i) / n2)
#     cos_teta_t = np.cos(teta_t)
#     return (n1 * cos_teta_i - n2 * cos_teta_t) / (n1 * cos_teta_i + n2 * cos_teta_t)
#
#
# def E_reflect_p(n1, n2, teta_i):
#     cos_teta_i = np.cos(teta_i)
#     teta_t = np.arcsin(n1 * np.sin(teta_i) / n2)
#     cos_teta_t = np.cos(teta_t)
#     return (n2 * cos_teta_i - n1 * cos_teta_t) / (n2 * cos_teta_i + n1 * cos_teta_t)
#
#
# def E_refract_s(n1, n2, teta_i):
#     cos_teta_i = np.cos(teta_i)
#     teta_t = np.arcsin(n1 * np.sin(teta_i) / n2)
#     cos_teta_t = np.cos(teta_t)
#     return (2 * n1 * cos_teta_i) / (n1 * cos_teta_i + n2 * cos_teta_t)
#
#
# def E_refract_p(n1, n2, teta_i):
#     cos_teta_i = np.cos(teta_i)
#     teta_t = np.arcsin(n1 * np.sin(teta_i) / n2)
#     cos_teta_t = np.cos(teta_t)
#     return (2 * n1 * cos_teta_i) / (n2 * cos_teta_i + n1 * cos_teta_t)


def E_refract_p(cos_alpha, cos_betta, E_p, n_1, n_2):
    return (2 * n_1 * cos_alpha) / (n_2 * cos_alpha + n_1 * cos_betta) * E_p


def E_refract_s(cos_alpha, cos_betta, E_s, n_1, n_2):
    return (2 * n_1 * cos_alpha) / (n_1 * cos_alpha + n_2 * cos_betta) * E_s


def E_reflect_p(cos_alpha, cos_betta, E_p, n_1, n_2):
    return (n_2 * cos_alpha - n_1 * cos_betta) / (n_2 * cos_alpha + n_1 * cos_betta) * E_p


def E_reflect_s(cos_alpha, cos_betta, E_s, n_1, n_2):
    return (n_1 * cos_alpha - n_2 * cos_betta) / (n_2 * cos_alpha + n_1 * cos_betta) * E_s