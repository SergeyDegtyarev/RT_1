import numpy as np


# вычисляет точку пересечения луча с плоскостью
def cross_surface(radius_vector_surface, normal_surface, radius_vector_ray, direction):
    if (normal_surface.dot(direction) == 0):
        return None
    answer = ((normal_surface.dot(radius_vector_surface - radius_vector_ray))
              / (normal_surface.dot(direction)))
    if answer < 0:
        return None
    return answer


def get_delta(r_v_sph, radius, r_v_ray, uni):
    delta = ((r_v_ray - r_v_sph).dot(uni)) ** 2 - ((r_v_ray - r_v_sph).dot(r_v_ray - r_v_sph)) + radius ** 2
    if delta < 0:
        return None
    else:
        return delta ** 0.5


# функция, возвращает длину луча до пересечения со сферой
def cross_sphere(r_v_sph, radius, r_v_ray, uni):
    delta = get_delta(r_v_sph, radius, r_v_ray, uni)
    if delta == None:
        return None, None
    elif delta == 0:
        answer = (0 - (r_v_ray - r_v_sph).dot(uni))
        if (answer < 0):
            return None, None
        return answer, None
    else:
        answer1 = -(r_v_ray - r_v_sph).dot(uni) + delta
        answer2 = -(r_v_ray - r_v_sph).dot(uni) - delta
        if (answer1 > 0) and (answer2 > 0):
            return answer1, answer2
        elif (answer1 > 0) and (answer2 < 0):
            return answer1, None
        elif (answer1 < 0) and (answer2 > 0):
            return answer2, None
        elif (answer1 < 0) and (answer2 < 0):
            return None, None

