from constants import n1, n2, k
import copy
import matplotlib.pyplot as plt
from classes import *
from cross import *
from laws import *
from draw import makeSurface, polar_ellipse, draw_ellipse
from polarization import *

rad_vec_sphere_1 = [-8, 0, 0]
rad_vec_sphere_2 = [8, 0, 0]
R1 = 10
R2 = 10

sphere1 = Sphere(R1, rad_vec_sphere_1)
sphere2 = Sphere(R2, rad_vec_sphere_2)

F1 = (n1 * R1 * R2) / (n2 - n1) / (R1 + R2)

print("Focus 1 ", F1)

thickness_lense = 1
radius_lense = np.sqrt(R1 ** 2 - (R1 - thickness_lense) ** 2)
source_light = []

radius_for_cycle = int(radius_lense) #для цикла for
#

rad_vec_sphere_3 = [-4, 0, 0]
rad_vec_sphere_4 = [12, 0, 0]
R3 = 10
R4 = 10

sphere3 = Sphere(R3, rad_vec_sphere_3)
sphere4 = Sphere(R4, rad_vec_sphere_4)

F2 = (n1 * R3 * R4) / (n2 - n1) / (R3 + R4)

print("Focus 2 ", F2)

# for phi in range (0, 360, 1):
#     r = 0.1
#     while r < radius_lense - 1:
#         source_light.append([-7, r * np.cos(phi * np.pi / 180), r * np.sin(phi * np.pi / 180)])
#         r += 1

# for i in range(-radius_for_cycle + 1, radius_for_cycle - 1, 1):
#     for j in range(-radius_for_cycle + 1, radius_for_cycle - 1, 1):
#         source_light.append([-7, i, j])

# edge = 1.5
#
# i = -edge
# while(i <= edge):
#     j = 10 - edge
#     while(j <= 10 + edge):
#         source_light.append([-10, i, j])
#         j += 0.1
#     i += 0.1

edge = 3

i = -edge
while(i <= edge):
    j = -edge
    while(j <= edge):
        source_light.append([-10, i, j])
        j += 0.17
    i += 0.17

# source_light.append([-10, 0, 0])
# source_light.append([-10, 0, 1])
# source_light.append([-10, 0, -1])

# source_light.append([-5, 0, 10])


# source_light = [[-7, 0, 0], [-7, 0, 1], [-7, 0, 2], [-7, 0, 3],[-7, 0, -1], [-7, 0, -2],[-7, 0, -3],
#                 [-7, 1, 0], [-7, 1, 1], [-7, 1, 2], [-7, 1, 3],[-7, 1, -1], [-7, 1, -2],[-7, 1, -3],
#                 [-7, -1, 0], [-7, -1, 1], [-7, -1, 2], [-7, -1, 3],[-7, -1, -1], [-7, -1, -2],[-7, -1, -3],
#                 [-7, 2, 0], [-7, 2, 1], [-7, 2, 2], [-7, 2, 3], [-7, 2, -1], [-7, 2, -2], [-7, 2, -3],
#                 [-7, 3, 0], [-7, 3, 1], [-7, 3, 2], [-7, 3, 3], [-7, 3, -1], [-7, 3, -2], [-7, 3, -3],
#                 [-7, -2, 0], [-7, -2, 1], [-7, -2, 2], [-7, -2, 3], [-7, -2, -1], [-7, -2, -2], [-7, -2, -3],
#                 [-7, -3, 0], [-7, -3, 1], [-7, -3, 2], [-7, -3, 3], [-7, -3, -1], [-7, -3, -2], [-7, -3, -3]]


print(len(source_light))

def var_1(path, optical_path, rays3):
    step = 0
    dispersion_prev = 0
    dispersion_for_graphic = []
    ind_for_dispersion = 0
    while(True):
        dispersion = 0 #дисперсия обнуляется на каждом шаге
        displs = 0.01 #смещение регистрирующей плоскости с каждым шагом на 0.1 оптического пути
        count = path #возможно лишняя строчка???
        point_register = [path[0] + displs * step / n1, 0, 0] #определяется положение регистрирующей плоскости на этом шаге
        normal_register = [1, 0, 0]
        register = Surface(normal_register, point_register) #строится регистрирующая плоскость на этом шаге
        ind = int(0) #счетчик чтоб по лучам прыгать
        optical_path_to_register = []
        cross_register_surface = []
        cross_register_points = []
        for op in optical_path:
            optical_path_to_register.append(op)
        for ray in rays3:
            cross_register = cross_surface(register.get_radius_vector(), register.get_normal(), ray.get_radius_vector(),
                                           ray.get_direction_vector()) # длина этого луча до пересечения плоскости на этом шаге
            cross_register_points.append(np.multiply(ray.get_direction_vector(), cross_register) + ray.get_radius_vector())
            cross_register_surface.append(cross_register)
            optical_path_to_register[ind] = optical_path[ind] + cross_register * n1 #оптический путь этого луча номером ind
            # dispersion.append(??????)
            ind += 1 #переключаем на следующий луч
        summ_of_op = 0 #сумму оптических лучей обнуляем, каждый раз заново будет считаться
        for optical_path_to_reg in optical_path_to_register:
            summ_of_op += optical_path_to_reg #суммируем все оптические длины лучей
        average = summ_of_op / len(optical_path_to_register) #теперь находим средний оптический путь на этом шаге (при этом положении регистрирующей плоскости)
        # ind = int(0)
        for ind in range(len(optical_path_to_register)):
            # dispersion += np.sqrt((cross_register_points[ind][1] ** 2 + cross_register_points[ind][2] ** 2 + (average - optical_path_to_register[ind]) ** 2) / (2 * len(optical_path_to_register))) #находим ско оптического пути для каждого положения плоскости
            dispersion += np.sqrt((cross_register_points[ind][1] ** 2 + cross_register_points[ind][2] ** 2) / (2 * len(optical_path_to_register))) #находим ско оптического пути для каждого положения плоскости
            # ind += 1
        # dispersion = np.sqrt(dispersion / (2 * len(optical_path_to_register)))
        dispersion_for_graphic.append(dispersion)
        if step == 0:
            step += 1
            dispersion_prev = dispersion
            continue
        print('STEP: ', step, ' DISPERSION: ', dispersion)

        step += 1
        if (dispersion_prev < dispersion):
            break
        dispersion_prev = dispersion
        ind_for_dispersion += 1
    # dispersion = dispersion_for_graphic[ind_for_dispersion - 1]
    print("POINT FOCUS: ", point_register)
    return (register, dispersion_for_graphic)

def var_2(optical_path, rays4):
    point_register = [0, 0, 0]
    normal_register = [1, 0, 0]
    register = Surface(normal_register, point_register)
    step = 0
    dispersion_prev = 0
    dispersion_for_graphic = []
    optical_path_to_point = []
    points = []
    # rays4 = []
    # rays4[:] = rays3[:]
    for a in rays4:
        points.append(a.get_radius_vector())
    for op in optical_path:
        optical_path_to_point.append(op)
    max_op = max(optical_path_to_point)
    list_of_distance = []
    while(True):
        flag = False
        while(flag == False):
            ind = 0
            flag = True
            for ray in rays4:
                if (optical_path_to_point[ind] < max_op):
                    points[ind] += ray.get_direction_vector() / 1000
                    optical_path_to_point[ind] += np.sqrt(ray.get_direction_vector()[0] ** 2 + ray.get_direction_vector()[1] ** 2 + ray.get_direction_vector()[2] ** 2) / 1000 * n1
                    flag = False
                ind += 1
        sum = [0, 0, 0]
        for i in range(len(points)):
            sum[0] += points[i][0]
            sum[1] += points[i][1]
            sum[2] += points[i][2]
        average_point = [sum[0] / len(points), sum[1] / len(points), sum[2] / len(points)]
        distance = 0
        for i in range(len(points)):
            distance += np.sqrt((average_point[0] - points[i][0]) ** 2 + (average_point[1] - points[i][1]) ** 2 + (average_point[2] - points[i][2]) ** 2)
        average_distance = distance / len(points)
        list_of_distance.append(average_distance)
        if(step > 0):
            if (list_of_distance[step] > list_of_distance[step - 1]):
                break
        step += 1
        max_op += 0.1
        point_register = average_point #определяется положение регистрирующей плоскости на этом шаге
        normal_register = [1, 0, 0]
        register = Surface(normal_register, point_register)
    # print("MAX", max_op)
    # print("OPTICAL PATH TO POINT", optical_path_to_point)
    # print("POINTS", points)
    # print("SUMMA", sum)
    # print("AVERAGE", average_point)
    # print("LIST_OF_DISTANCE", list_of_distance)
    # print("NORMA", register.get_normal())
    # print("POINT REG", register.get_radius_vector())
    plt.plot(list_of_distance)
    plt.show()
    # print("POINTSSSSSSSSS: ", points)
    ind = 0
    new_points = []
    E_of_rays = []
    for point in points:
        lamb = (average_point[0] - point[0]) / rays4[ind].get_direction_vector()[0]
        point = point + rays4[ind].get_direction_vector() * lamb
        new_points.append(point)
        E_of_rays.append(rays4[ind].get_E())
        ind += 1
    # print("POINTSSSSSSSSS: ", new_points)

    return register, list_of_distance, new_points, optical_path_to_point, E_of_rays


def focusing():
    rays1 = []
    rays2 = []
    rays3 = []
    rays7 = []
    rays8 = []
    optical_path = []
    optical_path_to_register = []
    path = []
    path_to_register = []
    cross_lens_1 = []
    cross_lens_2 = []
    cross_lens_3 = []
    cross_lens_4 = []
    cross_register_surface_points = []
    cross_register_surface = []
    ind = int(0)

    # for i in source_light:
    #     if random.randint(0, 30) < 10:
    #         rays1.append(Ray(i, [1, 0, 0], (1 / np.sqrt(2), 1j / np.sqrt(2))))
    #         ind += 1
    #     else: continue


    for i in source_light:
        # rays1.append(Ray(i, [0.707, 0, -0.707], (1, 0)))
        # rays1.append(Ray(i, [0.707, 0, -0.707], (1 / np.sqrt(2), 1j / np.sqrt(2))))
        # rays1.append(Ray(i, [0.45, 0, -0.89], (1 / np.sqrt(2), 1j / np.sqrt(2))))
        # rays1.append(Ray(i, [0.45, 0, -0.89], (1, 0)))
        # rays1.append(Ray(i, [1, 0, 0], np.array([1 / np.sqrt(2), 1j / np.sqrt(2)]))) #лучам присваивается исходный вектор Джонса
        # if ind % 2 == 0:
        rays1.append(Ray(i, [1, 0, 0], np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])))  # лучам присваивается исходный вектор Джонса


        # else:
        #     rays1.append(Ray(i, [1, 0, 0], np.array([1 / np.sqrt(2), -1j / np.sqrt(2)])))  # лучам присваивается исходный вектор Джонса

        ind += 1

    ###################################################ПЕРВОЕ ПЕРЕСЕЧЕНИЕ С ЛИНЗОЙ#######################################################
    for ray in rays1:
        cross1, cross2 = cross_sphere(rad_vec_sphere_2, R2, ray.get_radius_vector(), ray.get_direction_vector())
        if (cross1 < cross2):
            if cross1 == None: continue
            cross_lens_1.append(cross1)
            optical_path.append(cross1 * n1)
            path.append(cross1)
        else:
            if cross2 == None: continue
            cross_lens_1.append(cross2)
            optical_path.append(cross2 * n1)
            path.append(cross2)
        # print(cross_lens_1)

    tangents = []
    bitangents = []
    # for _ in range(len(rays1)):
    #     tangent.append([0, 1, 0])
    #     bitangent.append([0, 0, 1])
    ##########################################ВНУТРИ ЛИНЗЫ################################################################
    ind = int(0)
    for ray in rays1:
        dir_vec = ray.get_direction_vector()
        rad_vec = ray.get_radius_vector()
        normal_sphere = sphere2.get_normal_sphere(np.multiply(dir_vec, cross_lens_1[ind]) + rad_vec)
        E = ray.get_E()

        tan, bitan = basis_of_ray(dir_vec) #присвоение значений тангента и битангента
        tangents.append(tan)#они образуют тройку ортогональных векторов с вектором направления
        bitangents.append(bitan)

        s, p = raschet(dir_vec, normal_sphere, tangents[ind], bitangents[ind]) #находим векторы с и п (либо т и б либо по формулам)

        cos_alpha = np.fabs(np.dot(dir_vec, normal_sphere))
        new_dir_vec = law_refr(dir_vec, normal_sphere, n1, n2)
        cos_betta = np.fabs(np.dot(new_dir_vec, normal_sphere))

        E_t = ray.get_E()[0] #Е1
        E_b = ray.get_E()[1] #Е2

        # jv_s_p = Es_Ep(E, s, p, dir_vec)

        # Es = jv_s_p[1]
        # Ep = jv_s_p[2]

        Es = E_s(E_t, tangents[ind], s, E_b, bitangents[ind]) #проекция вектора Джонса на вектор с
        Ep = E_p(E_t, tangents[ind], E_b, bitangents[ind], p) #на вектор п

        E_refr_s = E_refract_s(cos_alpha, cos_betta, Es, n1, n2)#пересчет компонент по формулам френеля
        E_refr_p = E_refract_p(cos_alpha, cos_betta, Ep, n1, n2)

        tangents[ind], bitangents[ind] = new_tangent_bitangent(new_dir_vec, s)

        # переход в лаборатоную систему координат
        tangent_lab = [0, 1, 0]
        bitangent_lab = [0, 0, 1]
        Es_refr = E_s(E_refr_s, tangents[ind], tangent_lab, E_refr_p, bitangents[ind])
        Ep_refr = E_p(E_refr_s, tangents[ind], E_refr_p, bitangents[ind], bitangent_lab)

        # jv = np.array([Es_refr, Ep_refr])
        # np.linalg.norm(jv)
        jv = np.array([E_refr_s, E_refr_p])
        # np.linalg.norm(jv)

        rays2.append(Ray(np.multiply(dir_vec, cross_lens_1[ind]) + rad_vec, new_dir_vec, jv))

        ind += 1

    ##################################################ВТОРОЕ ПЕРЕСЕЧЕНИЕ С ЛИНЗОЙ#######################################################
    ind = int(0)
    for ray in rays2:
        cross1, cross2 = cross_sphere(rad_vec_sphere_1, R1, ray.get_radius_vector(), ray.get_direction_vector())
        if cross1 == None: continue
        cross_lens_2.append(cross1)
        optical_path[ind] += cross1 * n2
        path[ind] += cross1
        # print('OPTICAL PATH: ', optical_path[ind])
        ind += 1

    ##########################################ВЫХОД ИЗ ЛИНЗЫ################################################################
    ind = int(0)
    for ray in rays2:
        dir_vec = ray.get_direction_vector()
        rad_vec = ray.get_radius_vector()
        normal_sphere = sphere1.get_normal_sphere(np.multiply(dir_vec, cross_lens_2[ind]) + rad_vec)
        E = ray.get_E()

        s, p = raschet(dir_vec, normal_sphere, tangents[ind], bitangents[ind])

        cos_alpha = np.fabs(np.dot(dir_vec, normal_sphere))
        new_dir_vec = law_refr(dir_vec, normal_sphere, n2, n1)
        cos_betta = np.fabs(np.dot(new_dir_vec, normal_sphere))

        E_t = ray.get_E()[0]
        E_b = ray.get_E()[1]
        #
        # jv_s_p = Es_Ep(E, s, p, dir_vec)
        #
        # Es = jv_s_p[1]
        # Ep = jv_s_p[2]

        Es = E_s(E_t, tangents[ind], s, E_b, bitangents[ind])
        Ep = E_p(E_t, tangents[ind], E_b, bitangents[ind], p)

        E_refr_s = E_refract_s(cos_alpha, cos_betta, Es, n2, n1)
        E_refr_p = E_refract_p(cos_alpha, cos_betta, Ep, n2, n1)

        tangents[ind], bitangents[ind] = new_tangent_bitangent(new_dir_vec, s)

        # переход в лаборатоную систему координат
        tangent_lab = [0, 1, 0]
        bitangent_lab = [0, 0, 1]
        Es_refr = E_s(E_refr_s, tangents[ind], tangent_lab, E_refr_p, bitangents[ind])
        Ep_refr = E_p(E_refr_s, tangents[ind], E_refr_p, bitangents[ind], bitangent_lab)

        # jv = np.array([Es_refr, Ep_refr])
        jv = np.array([E_refr_s, E_refr_p])
        # np.linalg.norm(jv)

        rays3.append(Ray(np.multiply(dir_vec, cross_lens_2[ind]) + rad_vec, new_dir_vec, jv))

        ind += 1


    ind = int(0)
    for ray in rays3:
        cross3, cross4 = cross_sphere(rad_vec_sphere_4, R4, ray.get_radius_vector(), ray.get_direction_vector())
        if (cross3 < cross4):
            if cross3 == None: continue
            cross_lens_3.append(cross3)
            optical_path[ind] += cross3 * n1
            path[ind] += cross3
        else:
            if cross4 == None: continue
            cross_lens_3.append(cross4)
            optical_path[ind] += cross4 * n1
            path[ind] += cross4
        ind += 1


    ind = int(0)
    for ray in rays3:
        dir_vec = ray.get_direction_vector()
        rad_vec = ray.get_radius_vector()
        normal_sphere = sphere4.get_normal_sphere(np.multiply(dir_vec, cross_lens_3[ind]) + rad_vec)

        s, p = raschet(dir_vec, normal_sphere, tangents[ind], bitangents[ind]) #находим векторы с и п (либо т и б либо по формулам)

        cos_alpha = np.fabs(np.dot(dir_vec, normal_sphere))
        new_dir_vec = law_refr(dir_vec, normal_sphere, n1, n2)
        cos_betta = np.fabs(np.dot(new_dir_vec, normal_sphere))

        E_t = ray.get_E()[0] #Е1
        E_b = ray.get_E()[1] #Е2

        # jv_s_p = Es_Ep(E, s, p, dir_vec)

        # Es = jv_s_p[1]
        # Ep = jv_s_p[2]

        Es = E_s(E_t, tangents[ind], s, E_b, bitangents[ind]) #проекция вектора Джонса на вектор с
        Ep = E_p(E_t, tangents[ind], E_b, bitangents[ind], p) #на вектор п

        E_refr_s = E_refract_s(cos_alpha, cos_betta, Es, n1, n2)#пересчет компонент по формулам френеля
        E_refr_p = E_refract_p(cos_alpha, cos_betta, Ep, n1, n2)

        tangents[ind], bitangents[ind] = new_tangent_bitangent(new_dir_vec, s)

        # переход в лаборатоную систему координат
        tangent_lab = [0, 1, 0]
        bitangent_lab = [0, 0, 1]
        Es_refr = E_s(E_refr_s, tangents[ind], tangent_lab, E_refr_p, bitangents[ind])
        Ep_refr = E_p(E_refr_s, tangents[ind], E_refr_p, bitangents[ind], bitangent_lab)

        # jv = np.array([Es_refr, Ep_refr])
        # np.linalg.norm(jv)
        jv = np.array([E_refr_s, E_refr_p])
        # np.linalg.norm(jv)

        rays7.append(Ray(np.multiply(dir_vec, cross_lens_3[ind]) + rad_vec, new_dir_vec, jv))

        ind += 1

    ##################################################ВТОРОЕ ПЕРЕСЕЧЕНИЕ С ЛИНЗОЙ#######################################################
    ind = int(0)
    for ray in rays7:
        cross3, cross4 = cross_sphere(rad_vec_sphere_3, R3, ray.get_radius_vector(), ray.get_direction_vector())
        if cross3 == None: continue
        cross_lens_4.append(cross3)
        optical_path[ind] += cross3 * n2
        path[ind] += cross3
        # print('OPTICAL PATH: ', optical_path[ind])
        ind += 1

    ##########################################ВЫХОД ИЗ ЛИНЗЫ################################################################
    ind = int(0)
    for ray in rays7:
        dir_vec = ray.get_direction_vector()
        rad_vec = ray.get_radius_vector()
        normal_sphere = sphere3.get_normal_sphere(np.multiply(dir_vec, cross_lens_4[ind]) + rad_vec)

        s, p = raschet(dir_vec, normal_sphere, tangents[ind], bitangents[ind])

        cos_alpha = np.fabs(np.dot(dir_vec, normal_sphere))
        new_dir_vec = law_refr(dir_vec, normal_sphere, n2, n1)
        cos_betta = np.fabs(np.dot(new_dir_vec, normal_sphere))

        E_t = ray.get_E()[0]
        E_b = ray.get_E()[1]
        #
        # jv_s_p = Es_Ep(E, s, p, dir_vec)
        #
        # Es = jv_s_p[1]
        # Ep = jv_s_p[2]

        Es = E_s(E_t, tangents[ind], s, E_b, bitangents[ind])
        Ep = E_p(E_t, tangents[ind], E_b, bitangents[ind], p)

        E_refr_s = E_refract_s(cos_alpha, cos_betta, Es, n2, n1)
        E_refr_p = E_refract_p(cos_alpha, cos_betta, Ep, n2, n1)

        # переход в лаборатоную систему координат
        tangent_lab = [0, 1, 0]
        bitangent_lab = [0, 0, 1]
        Es_refr = E_s(E_refr_s, tangents[ind], tangent_lab, E_refr_p, bitangents[ind])
        Ep_refr = E_p(E_refr_s, tangents[ind], E_refr_p, bitangents[ind], bitangent_lab)

        # jv = np.array([Es_refr, Ep_refr])
        jv = np.array([E_refr_s, E_refr_p])
        # np.linalg.norm(jv)

        rays8.append(Ray(np.multiply(dir_vec, cross_lens_4[ind]) + rad_vec, new_dir_vec, jv))

        ind += 1

    ##############################################РЕГИСТРИРУЮЩАЯ ПОВЕРХНОСТЬ############################################################
    ind = int(0)
    # register = Surface([1, 0, 0], [9.1, 0, 0])
    # var_2()
    rays5 = copy.deepcopy(rays8)
    rays4 = copy.deepcopy(rays8)
    # register, list_for_graphic = var_1(path, optical_path, rays8)
    register, list_for_graphic, points, optical_path_to_point, E_of_rays = var_2(optical_path, rays4)

    energy = []
    ind = 0
    for ray in rays5:
        vec = points[ind] - ray.get_radius_vector()
        length = np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        optical_path_this_ray = length * n1 + optical_path[ind]
        energy.append(np.exp((1j + 0) * k * optical_path_this_ray))
        ind += 0

    N = 171
    matrix = np.zeros([N, N])
    matrix_energy = np.zeros([N, N], dtype = complex)
    matrix_polyariz = np.zeros([N, N, 2], dtype = complex)
    matrix_energy_phase = np.zeros([N, N])
    matrix_energy_amplitude = np.zeros([N, N])
    # register_size = 0.2
    register_size = 1
    pixel_size = register_size / N
    value = 0.5
    ind = 0
    for point, E in zip(points, E_of_rays):
        if value < 0:
            value *= -1
        if point[1] < 0:
            value *= -1
        n = int(N / 2) + int(point[1] / pixel_size + value)
        if value < 0:
            value *= -1
        if point[2] < 0:
            value *= -1
        m = int(N / 2) + int(point[2] / pixel_size + value)
        if n >= N or m >= N or n < 0 or m < 0: continue
        matrix[n, m] += 1
        matrix_polyariz[n, m, 0] += E[0]
        matrix_polyariz[n, m, 1] += E[1]
        matrix_energy[n, m] += energy[ind]
        ind += 1

    for i in range(N):
        for j in range(N):
            jv = np.array([matrix_polyariz[i, j, 0], matrix_polyariz[i, j, 1]])
            np.linalg.norm(jv)
            matrix_polyariz[i, j, 0] = jv[0]
            matrix_polyariz[i, j, 1] = jv[1]


    print()
    # print(matrix_polyariz[pixel_row, pixel_col, 0], " ", matrix_polyariz[pixel_row, pixel_col, 1])
    # jv = np.array([matrix_polyariz[pixel_row, pixel_col, 0], matrix_polyariz[pixel_row, pixel_col, 1]])
    # polar_ellipse(jv, 'title_figure', 'title_ray')
    # draw_ellipse(matrix_polyariz[int(N / 2) + 10, int(N / 2) + 10, 0], matrix_polyariz[int(N / 2) + 10, int(N / 2) + 10, 1])

    ind = 0
    for n in range(len(matrix_energy[0])):
        for m in range(len(matrix_energy[1])):
            matrix_energy_phase[n][m] = np.angle(matrix_energy[n][m])
            matrix_energy_amplitude[n][m] = np.abs(matrix_energy[n][m])


    Y = np.linspace(-register_size/2, register_size/2, N)
    X = []
    X1 = []
    X2 = []
    sum = 0
    for i in range(N):
        val = 0
        val_energy_d = 0
        val_energy_m = 0
        for j in range(N):
            val += matrix[j][i]
            val_energy_d += np.abs(matrix_energy[j][i])
            val_energy_m += np.angle(matrix_energy[j][i])
        X.append(val)
        X1.append(val_energy_d)
        X2.append(val_energy_m)
        sum += val

    # print(matrix_energy)

    max = 0
    m = 0
    n = 0
    for i in range(N):
        for j in range(N):
            if matrix[i, j] >= max:
                max = matrix[i, j]
                n = i
                m = j


    pixel_row = n
    pixel_col = m

    print("M N !@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@ ", m, " ", n)
    # Set general font size
    plt.rcParams['font.size'] = '7'

    plt.figure(figsize=(4, 3))

    # ind = int(1)
    # for i in range(5):
    #     for j in range(5):
    #         plt.subplot(5, 5, ind)
    #         jv = np.array([rays2[ind].get_E()[0], rays2[ind].get_E()[1]])
    #         polar_ellipse(jv, str(pixel_row + i) + " " + str(pixel_col + j))
    #         # draw_ellipse(rays3[ind].get_E()[0], rays3[ind].get_E()[1])
    #         plt.plot()
    #         if ind >= 24: break
    #         ind += 1
    # plt.show()


    ind = int(1)
    for i in range(-2, 3, 1):
        for j in range(-2, 3, 1):
            plt.subplot(5, 5, ind)
            jv = np.array([matrix_polyariz[pixel_row + i, pixel_col + j, 0], matrix_polyariz[pixel_row + i , pixel_col + j, 1]])
            polar_ellipse(jv, str(matrix[pixel_row + i, pixel_row + j]))
            # draw_ellipse(matrix_polyariz[pixel_row + i, pixel_col + j, 0], matrix_polyariz[pixel_row + i , pixel_col + j, 1])
            # , str(pixel_row + i) + " " + str(pixel_col + j)
            # plt.set_title("Эллипсы поляризации после преломления в линзе пучка лучей с линейной поляризацией")
            plt.plot()
            ind += 1
    plt.show()

    print(sum)
    plt.plot(Y, X)
    plt.xlabel('Значение по оси Z')
    plt.ylabel('Количество лучей, шт')
    plt.title('Проекция количества лучей в матрице на ось Y')
    plt.grid(True)

    plt.show()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
    p1 = axs[0].imshow(matrix_energy_amplitude, cmap='gist_stern', aspect='equal', origin="lower")
    fig.colorbar(p1, ax=axs[0])
    p2 = axs[1].imshow(matrix_energy_phase, cmap='plasma', aspect='equal', origin="lower")
    fig.colorbar(p2, ax=axs[1])
    plt.subplot(1, 2, 1)
    plt.imshow(matrix_energy_amplitude)
    plt.title("Амплитуда")
    plt.subplot(1, 2, 2)
    plt.title("Фаза")
    plt.imshow(matrix_energy_phase)
    plt.figure()
    plt.imshow(matrix)
    plt.plot(Y, X1)
    plt.show()
    plt.plot(Y, X2)
    plt.show()

    for ray in rays8:
        cross_register = cross_surface(register.get_radius_vector(), register.get_normal(), ray.get_radius_vector(), ray.get_direction_vector())
        if cross_register == None: continue
        cross_register_surface.append(cross_register)
        cross_register_surface_points.append(np.multiply(ray.get_direction_vector(), cross_register) + ray.get_radius_vector())
        ind += 1

    # plt.plot(list_for_graphic)
    # plt.show()

    # plt.plot(energy)
    # plt.show()

    print("0000000000000000000000000000000000000000000000000000000000000000000000000000")
    print(cross_register_surface_points)

    y = []
    z = []
    for i in range(len(cross_register_surface_points)):
        y.append(cross_register_surface_points[i][1])
        z.append(cross_register_surface_points[i][2])


    plt.scatter(y, z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([-10, 40])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-6, 6])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    x, y, z = makeSurface(register.get_normal(), register.get_radius_vector())
    ax.plot_surface(x, y, z)
    print('**********************************************************************************')
    print(register.get_radius_vector())

    # start_incident = myRay.get_radius_vector()
    # incident = myRay.get_direction_vector() * cross_point_surface
    # axes.quiver(start_incident[0], start_incident[1], start_incident[2], incident[0], incident[1], incident[2], color = 'r')
    # start_refr_refl = finish_incident_surface
    # normal = 5 * mySurf.get_normal()
    # axes.quiver(start_refr_refl[0], start_refr_refl[1], start_refr_refl[2], normal[0], normal[1], normal[2])
    # refl = 5 * law_refl(myRay.get_direction_vector(), mySurf.get_normal())
    # axes.quiver(start_refr_refl[0], start_refr_refl[1], start_refr_refl[2],
    #             refl[0], refl[1], refl[2], color='g')
    # refr = 5 * law_refr(myRay.get_direction_vector(), mySurf.get_normal(), n1,
    #                     n2)
    # axes.quiver(start_refr_refl[0], start_refr_refl[1], start_refr_refl[2],
    #             refr[0], refr[1], refr[2], color='y')
    for i in range(len(rays7)):
        if i >= len(cross_register_surface): break
        ax.quiver(rays1[i].get_radius_vector()[0], rays1[i].get_radius_vector()[1], rays1[i].get_radius_vector()[2],
                  rays1[i].get_direction_vector()[0] * cross_lens_1[i], rays1[i].get_direction_vector()[1] * cross_lens_1[i], rays1[i].get_direction_vector()[2] * cross_lens_1[i], color = 'r', arrow_length_ratio = 0, linewidths = 1)

        ax.quiver(rays2[i].get_radius_vector()[0], rays2[i].get_radius_vector()[1], rays2[i].get_radius_vector()[2],
                  rays2[i].get_direction_vector()[0] * cross_lens_2[i], rays2[i].get_direction_vector()[1] * cross_lens_2[i], rays2[i].get_direction_vector()[2] * cross_lens_2[i], color = 'r', arrow_length_ratio = 0, linewidths = 1)

        ax.quiver(rays3[i].get_radius_vector()[0], rays3[i].get_radius_vector()[1], rays3[i].get_radius_vector()[2],
                  rays3[i].get_direction_vector()[0] * cross_lens_3[i], rays3[i].get_direction_vector()[1] * cross_lens_3[i], rays3[i].get_direction_vector()[2] * cross_lens_3[i], color = 'r', arrow_length_ratio = 0, linewidths = 1)

        ax.quiver(rays7[i].get_radius_vector()[0], rays7[i].get_radius_vector()[1], rays7[i].get_radius_vector()[2],
                  rays7[i].get_direction_vector()[0] * cross_lens_4[i], rays7[i].get_direction_vector()[1] * cross_lens_4[i], rays7[i].get_direction_vector()[2] * cross_lens_4[i], color = 'r', arrow_length_ratio = 0, linewidths = 1)

        ax.quiver(rays8[i].get_radius_vector()[0], rays8[i].get_radius_vector()[1], rays8[i].get_radius_vector()[2],
                  rays8[i].get_direction_vector()[0] * cross_register_surface[i], rays8[i].get_direction_vector()[1] * cross_register_surface[i], rays8[i].get_direction_vector()[2] * cross_register_surface[i], color = 'r', arrow_length_ratio = 0, linewidths = 1)

    Number = int(20)

    u = np.linspace(0, 2 * np.pi, Number)
    v = np.linspace(0, np.pi, Number)

    x1 = sphere1.get_radius_vector()[0] + sphere1.get_radius() * np.outer(np.cos(u), np.sin(v))
    y1 = sphere1.get_radius_vector()[1] + sphere1.get_radius() * np.outer(np.sin(u), np.sin(v))
    z1 = sphere1.get_radius_vector()[2] + sphere1.get_radius() * np.outer(np.ones(np.size(u)), np.cos(v))

    x2 = sphere2.get_radius_vector()[0] + sphere2.get_radius() * np.outer(np.cos(u), np.sin(v))
    y2 = sphere2.get_radius_vector()[1] + sphere2.get_radius() * np.outer(np.sin(u), np.sin(v))
    z2 = sphere2.get_radius_vector()[2] + sphere2.get_radius() * np.outer(np.ones(np.size(u)), np.cos(v))

    for i in range(len(x1)):
        for j in range(len(x1[i])):
            length1 = np.sqrt((x1[i][j] - sphere1.get_radius_vector()[0]) ** 2 +
                              (y1[i][j] - sphere1.get_radius_vector()[1]) ** 2 +
                              (z1[i][j] - sphere1.get_radius_vector()[2]) ** 2)
            length2 = np.sqrt((x1[i][j] - sphere2.get_radius_vector()[0]) ** 2 +
                              (y1[i][j] - sphere2.get_radius_vector()[1]) ** 2 +
                              (z1[i][j] - sphere2.get_radius_vector()[2]) ** 2)
            if not(length1 == R1 and length2 <= R2):
                x1[i][j] = 0
                y1[i][j] = 0
                z1[i][j] = 0

            length1 = np.sqrt((x2[i][j] - sphere1.get_radius_vector()[0]) ** 2 +
                              (y2[i][j] - sphere1.get_radius_vector()[1]) ** 2 +
                              (z2[i][j] - sphere1.get_radius_vector()[2]) ** 2)
            length2 = np.sqrt((x2[i][j] - sphere2.get_radius_vector()[0]) ** 2 +
                              (y2[i][j] - sphere2.get_radius_vector()[1]) ** 2 +
                              (z2[i][j] - sphere2.get_radius_vector()[2]) ** 2)
            if not(length2 == R2 and length1 <= R1):
                x2[i][j] = 0
                y2[i][j] = 0
                z2[i][j] = 0

    ax.plot_wireframe(x1, y1, z1, rstride=1, cstride=1, color='y', alpha=0.7)
    ax.plot_wireframe(x2, y2, z2, rstride=1, cstride=1, color='y', alpha=0.7)
    #
    x3 = sphere3.get_radius_vector()[0] + sphere3.get_radius() * np.outer(np.cos(u), np.sin(v))
    y3 = sphere3.get_radius_vector()[1] + sphere3.get_radius() * np.outer(np.sin(u), np.sin(v))
    z3 = sphere3.get_radius_vector()[2] + sphere3.get_radius() * np.outer(np.ones(np.size(u)), np.cos(v))

    x4 = sphere4.get_radius_vector()[0] + sphere4.get_radius() * np.outer(np.cos(u), np.sin(v))
    y4 = sphere4.get_radius_vector()[1] + sphere4.get_radius() * np.outer(np.sin(u), np.sin(v))
    z4 = sphere4.get_radius_vector()[2] + sphere4.get_radius() * np.outer(np.ones(np.size(u)), np.cos(v))

    for i in range(len(x3)):
        for j in range(len(x3[i])):
            length3 = np.sqrt((x3[i][j] - sphere3.get_radius_vector()[0]) ** 2 +
                              (y3[i][j] - sphere3.get_radius_vector()[1]) ** 2 +
                              (z3[i][j] - sphere3.get_radius_vector()[2]) ** 2)
            length4 = np.sqrt((x3[i][j] - sphere4.get_radius_vector()[0]) ** 2 +
                              (y3[i][j] - sphere4.get_radius_vector()[1]) ** 2 +
                              (z3[i][j] - sphere4.get_radius_vector()[2]) ** 2)
            if not(length3 == R3 and length4 <= R4):
                x3[i][j] = 4
                y3[i][j] = 0
                z3[i][j] = 0

            length3 = np.sqrt((x4[i][j] - sphere3.get_radius_vector()[0]) ** 2 +
                              (y4[i][j] - sphere3.get_radius_vector()[1]) ** 2 +
                              (z4[i][j] - sphere3.get_radius_vector()[2]) ** 2)
            length4 = np.sqrt((x4[i][j] - sphere4.get_radius_vector()[0]) ** 2 +
                              (y4[i][j] - sphere4.get_radius_vector()[1]) ** 2 +
                              (z4[i][j] - sphere4.get_radius_vector()[2]) ** 2)
            if not(length4 == R4 and length3 <= R3):
                x4[i][j] = 4
                y4[i][j] = 0
                z4[i][j] = 0

    ax.plot_wireframe(x3, y3, z3, rstride=1, cstride=1, color='y', alpha=0.7)
    ax.plot_wireframe(x4, y4, z4, rstride=1, cstride=1, color='y', alpha=0.7)
