import numpy as np

class Sphere:
    'Это класс сферы'

    def __init__(self, radius, radius_vector):  # инициализация сферы
        self.radius = radius
        self.radius_vector = np.array(radius_vector)

    def get_radius(self):  # функция которая возвращает радиус сферы
        return self.radius

    def get_radius_vector(self):  # возвращает радиус-вектор центра сферы
        return (self.radius_vector)

    def set_radius(self, radius):  # устанавливает радиус
        self.radius = radius

    def set_radius_vector(self, radius_vector):  # задает радиус-вектор центра
        self.radius_vector = np.array(radius_vector)

    def get_normal_sphere(self, cross_point):  # возвращает нормаль к сфере
        length = ((cross_point - self.get_radius_vector()).dot(cross_point -
                                                               self.get_radius_vector())) ** 0.5
        return np.divide((cross_point - self.get_radius_vector()), length)


class Surface:
    'Это класс плоскоть'

    def __init__(self, normal, point):  # инициализация плоскости
        self.normal = np.array(normal)
        self.point = np.array(point)
        length = ((self.normal).dot(self.normal)) ** 0.5
        if (length != 1):
            self.normal = np.divide(self.normal, length)

    def get_normal(self):  # возвращает нормаль к плоскости
        return self.normal

    def get_radius_vector(self):  # возвращает радиус-вектор к точке
        return self.point


class Ray:
    'Это класс луч'

    def __init__(self, rad_vec, dir_vec, E):  # инициализация луча
        self.origin = np.array(rad_vec)
        self.dir_vec = np.array(dir_vec)
        self.E = E
        length = ((self.dir_vec).dot(self.dir_vec)) ** 0.5
        if (length != 1):
            self.dir_vec = np.divide(self.dir_vec, length)

    def get_radius_vector(self):  # возвращает радиус-вектор точки начала луча
        return self.origin

    def get_direction_vector(self):  # возвращает единичный вектор направления луча
        return self.dir_vec

    def get_E(self):
        return self.E

