import cmath

import matplotlib.pyplot as plt
from constants import n1, n2
import numpy as np


def makeSurface(normal_surface, point_surface):
    D = normal_surface[0] * point_surface[0] + normal_surface[1] * point_surface[1] + normal_surface[2] * point_surface[2]
    if (normal_surface[0] == 0 and normal_surface[2] == 0):
        # Строим сетку в интервале от -10 до 10, имеющую 100 отсчетов по обоим координатам
        z = np.linspace(-10, 10, 100)
        x = np.linspace(-10, 10, 100)
        # Создаем двумерную матрицу-сетку
        zgrid, xgrid = np.meshgrid(z, x)
        # В узлах рассчитываем значение функции
        y = (D - normal_surface[0] * xgrid - normal_surface[2] * zgrid) / normal_surface[1]
        return xgrid, y, zgrid
    elif (normal_surface[0] == 0 and normal_surface[1] == 0):
        # Строим сетку в интервале от -10 до 10, имеющую 100 отсчетов по обоим координатам
        y = np.linspace(-10, 10, 100)
        x = np.linspace(-10, 10, 100)
        # Создаем двумерную матрицу-сетку
        ygrid, xgrid = np.meshgrid(y, x)
        # В узлах рассчитываем значение функции
        z = (D - normal_surface[0] * xgrid - normal_surface[1] * ygrid) /\
            normal_surface[2]
        return xgrid, ygrid, z
    else:
        z = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        # Создаем двумерную матрицу-сетку
        zgrid, ygrid = np.meshgrid(z, y)
        # В узлах рассчитываем значение функции
        x = (D - normal_surface[1] * ygrid - normal_surface[2] * zgrid) /\
            normal_surface[0]
        return x, ygrid, zgrid


def draw_energy_graphics(ind1, ind2, r_s, r_p, t_s, t_p):
    fig = plt.figure()
    axes = fig.add_subplot()
    if ind1 > ind2:
        t = np.linspace(0, np.pi / 2, 1000)
        y = np.fabs(r_s(n1, n2, t)) ** 2
        line_r_s = plt.plot(t, y, linestyle='--', color='darkblue')
        y = np.fabs(r_p(n1, n2, t)) ** 2
        line_r_p = plt.plot(t, y, linestyle='--', color='darkred')
        y = (np.fabs(t_s(n1, n2, t))) ** 2 * n2 * np.cos(np.arcsin((n1 *
                                                                    np.sin(t) / n2))) / (n1 * np.cos(t))
        line_t_s = plt.plot(t, y, color='darkblue')
        y = (np.fabs(t_p(n1, n2, t))) ** 2 * n2 * np.cos(np.arcsin((n1 *
                                                                    np.sin(t) / n2))) / (n1 * np.cos(t))
        line_t_p = plt.plot(t, y, color='darkred')
        plt.grid(color='lightgray', linestyle='--')
    else:
        t = np.linspace(0, np.arsin(ind1 / ind2) - 0.001, 1000)
        y = np.fabs(r_s(n1, n2, t)) ** 2
        line_r_s = plt.plot(t, y, linestyle='--', color='darkblue')
        y = np.fabs(r_p(n1, n2, t)) ** 2
        line_r_p = plt.plot(t, y, linestyle='--', color='darkred')
        y = (np.fabs(t_s(n1, n2, t))) ** 2 * n2 * np.cos(np.arcsin((n1 *
                                                                    np.sin(t) / n2))) / (n1 * np.cos(t))
        line_t_s = plt.plot(t, y, color='darkblue')
        y = (np.fabs(t_p(n1, n2, t))) ** 2 * n2 * np.cos(np.arcsin((n1 *
                                                                    np.sin(t) / n2))) / (n1 * np.cos(t))
        line_t_p = plt.plot(t, y, color='darkred')
        t1 = np.linspace(np.arcsin(ind1 / ind2), np.pi / 2, 1000)
        y = t1 ** 0
        line_r_s_2 = plt.plot(t1, y, linestyle='--', color='darkblue')
        line_r_p_2 = plt.plot(t1, y, linestyle='--', color='darkred')
        y = t1 * 0
        line_t_s = plt.plot(t1, y, color='darkblue')
        line_t_p = plt.plot(t1, y, color='darkred')
        axes.axvline(x=np.arcsin(ind1 / ind2), color='darkblue')
        axes.axvline(x=np.arcsin(ind1 / ind2), color='darkred')
        plt.grid(color='lightgray', linestyle='--')
    axes.set_xlim([0, np.pi / 2])
    axes.set_ylim([0, 1])
    plt.show()


# функция построения и отрисовки эллипса поляризации
def draw_ellipse(E1, E2):
    print('E1 ', E1)
    print('E2 ', E2)
    # fig = plt.figure()
    # axes = fig.add_subplot()
    u = 0  # x-position of the center
    v = 0  # y-position of the center
    amplitude_x = abs(E1)
    amplitude_y = abs(E2)
    gamma = np.angle(E1) - np.angle(E2)
    amplitude = np.sqrt(amplitude_x ** 2 + amplitude_y ** 2)
    if (amplitude_x ** 2 - amplitude_y ** 2) == 0:
        alpha = 0
    else:
        alpha = (np.arctan((2 * amplitude_x * amplitude_y * np.cos(gamma)) /
                           (amplitude_x ** 2 - amplitude_y ** 2))) / 2
    ellipticity = np.tan(
        0.5 * np.arcsin(np.sin(gamma) * 2 * amplitude_x * amplitude_y / (amplitude_x ** 2 + amplitude_y ** 2)))
    beta = np.arctan(ellipticity)
    a = amplitude * np.cos(beta)  # radius on the x-axis
    b = amplitude * np.sin(beta)  # radius on the y-axis
    t_rot = alpha  # rotation angle
    t = np.linspace(0, 2 * np.pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    # u,v removed to keep the same center location
    R_rot = np.array([[np.cos(t_rot), -np.sin(t_rot)], [np.sin(t_rot), np.cos(t_rot)]])
    # 2-D rotation matrix
    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
    line = plt.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], 'darkorange')[0]  #rotated ellipse
    plt.grid(color='lightgray', linestyle='--')
    if np.angle(E1) > np.angle(E2):
        add_arrow(line, direction="right")
    else:
        add_arrow(line, direction="left")
    #
    # axes.set_xlim([-2, 2])
    # axes.set_ylim([-2, 2])
    q = 3
    plt.xlim(-q, q)
    plt.ylim(-q, q)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("X")
    plt.ylabel("Y")
    # axes.set_xlabel('X')
    # axes.set_ylabel('Y')

    # plt.show()
    # return plt



def add_arrow(line, position=None, direction='right', size=15, color=None):
    if color is None:
        color = line.get_color()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    if position is None:
        position = xdata.mean()
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1
    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size
                       )



def alphabeta_to_djones(alpha, beta, amp):
    jv = [1+1j, 1+1j]
    jv[0] = np.cos(alpha)*np.cos(beta)-1j*np.sin(alpha)*np.sin(beta)
    jv[1] = np.sin(alpha)*np.cos(beta)+1j*np.cos(alpha)*np.sin(beta)
    jv = jv * amp
    return jv


def polar_ellipse(jv_new, title_figure):
    A = np.linalg.norm(jv_new)
    if A == 0:  # энергия равна 0, рисуем жирную точку в центре
        plt.plot(0, 0, '*', color='blue')
        x = [0, 0]
        y = [0, 0]

    elif jv_new[0] == 0:  # рисуем вертикальную линию, поляризация горизонтальная
        x = [0, 0]
        y = [-A, A]

    elif jv_new[1] == 0:  # рисуем горизонтальную линию, поляризация горизонтальная
        x = [-A, A]
        y = [0, 0]


    else:  # поляризация линейная наклонная, круговая, эллиптическая
        chi = jv_new[1] / jv_new[0]
        if np.imag(chi) == 0.:  # поляризация линейная наклонная 4.1.
            if np.abs(chi)==1: # поляризация линейная наклонная под углом +-45 градусов 4.1.1

                if np.real(chi) > 0:# поляризация линейная наклонная под углом +45 градусов 4.1.1.1
                    x = [-A / np.sqrt(2), A / np.sqrt(2)]
                    y = [-A / np.sqrt(2), A / np.sqrt(2)]

                else: # поляризация линейная наклонная под углом -45 градусов 4.1.1.2
                    x = [-A / np.sqrt(2), A / np.sqrt(2)]
                    y = [A / np.sqrt(2), -A / np.sqrt(2)]

            else: # поляризация линейная наклонная под углом alpha 4.1.2
                alpha = np.arctan2(np.abs(jv_new[1]), np.abs(jv_new[0]))
                if cmath.phase(chi) != 0.:
                    alpha = -alpha

                x = [-A * np.cos(alpha), A * np.cos(alpha)]
                y = [-A * np.sin(alpha), A * np.sin(alpha)]
                print(cmath.phase(-1), A, chi, alpha)

        else: #поляризация круговая или эллиптическая
            chi = jv_new[1] / jv_new[0]
            if np.abs(chi) == 1.0:
                alpha = np.pi/4
            elif np.abs(chi) < 1.0:
                tga = 2 * np.real(chi) / (1 - np.abs(chi) **  2)
                alpha = np.arctan(tga) / 2
            else:
                tga = 2 * np.real(chi) / (1 - np.abs(chi) ** 2)
                alpha = (np.arctan(tga)+np.pi) / 2

            sinb = 2 * np.imag(chi) / (1 + np.abs(chi) ** 2)
            beta = np.arcsin(sinb) / 2
            a = A * np.cos(beta)
            b = A * np.sin(beta)

            # Далее операции по построению эллипса поляризации
            t = np.linspace(0, 360, 360)

            if b > a:
                r = b
            else:
                r = a
            x = r * np.cos(np.radians(t))
            y = r * np.sin(np.radians(t))
            plt.plot(x, y, '--', color='blue')
            x1 = a * np.cos(np.radians(t))
            y1 = b * np.sin(np.radians(t))
            x = x1 * np.cos(alpha) - y1 * np.sin(alpha)
            y = x1 * np.sin(alpha) + y1 * np.cos(alpha)
    plt.plot(x, y, '-', color='orange')

    plt.axis('equal')
    plt.title(title_figure, fontsize=8)
    plt.xlabel('x', fontsize=5)
    plt.ylabel('y', fontsize=5)
    plt.grid()



def jones_vector_transform(jv, t, b, s, p):
    jv_new = [1+1j, 1+1j]
    jv_new[0] = jv[0] * np.dot(t, s) + jv[1] * np.dot(b, s)
    jv_new[1] = jv[0] * np.dot(t, p) + jv[1] * np.dot(b, p)
    return jv_new

# def rotate_vector(t,alpha):
#     s = [1, 0]
#     s[0] = np.cos(alpha)*t[0]-np.sin(alpha)*t[1]
#     s[1] = np.sin(alpha)*t[0]+np.cos(alpha)*t[1]
#     return s

    #
    # for alpha in range(0, 360, 10):
    #     print(alpha)
    #     s = rotate_vector(t, np.pi*alpha/180)
    #     p = rotate_vector(b, np.pi*alpha/180)
    #     print(polar_ellipse(jones_vector_transform([10, 1j], t, b, s, p), 'title_figure', 'title_ray'))
    # plt.show()