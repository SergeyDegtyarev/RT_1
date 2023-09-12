from PIL import Image
from classes import *
from cross import *
from laws import *
from draw import *
from focus import focusing
from polarization import *
import locale
locale.setlocale(locale.LC_ALL, "Portuguese_Brazil.1252")
import matplotlib as mpl
mpl.rcParams['axes.formatter.use_locale'] = True


#print("Enter origin of ray: ")
a1, a2, a3 = [float(s) for s in input().split()]
rad_vec_ray = [a1, a2, a3]
print("Enter direction vector of ray: ")
b1, b2, b3 = [float(s) for s in input().split()]
direction_vector_ray = [b1, b2, b3]
print("Enter center of sphere: ")
c1, c2, c3 = [float(s) for s in input().split()]
rad_vec_sphere = [c1, c2, c3]
radius_sphere = float(input("Enter radius: "))
print("Enter point of surface: ")
d1, d2, d3 = [float(s) for s in input().split()]
point_surface = [d1, d2, d3]
print("Enter normal of surface: ")
q1, q2, q3 = [float(s) for s in input().split()]
normal_surface = [q1, q2, q3]
print("Enter E1, E2: ")
E1, E2 = [complex(s) for s in input().split()]
jones_vector = np.array([E1, E2])

mySh = Sphere(radius_sphere, rad_vec_sphere)
myRay = Ray(rad_vec_ray, direction_vector_ray, jones_vector)
mySurf = Surface(normal_surface, point_surface)

cross_point_surface = cross_surface(mySurf.get_radius_vector(),
                                    mySurf.get_normal(), myRay.get_radius_vector(), myRay.get_direction_vector())
if (cross_point_surface == None):
    print("луч не пересекает плоскость")
else:
    print("cross_surface: ", cross_point_surface)
    finish_incident_surface = np.multiply(myRay.get_direction_vector(),
                                          cross_point_surface) + myRay.get_radius_vector()
    print("cross_point_surface: ", finish_incident_surface)

cross_point_sphere = cross_sphere(mySh.get_radius_vector(),
                                  mySh.get_radius(), myRay.get_radius_vector(), myRay.get_direction_vector())
if (cross_point_sphere == None):
    print("луч не пересекает сферу")
else:
    print("cross_sphere: ", cross_point_sphere)
    finish_incident_sphere1 = np.multiply(myRay.get_direction_vector(),
                                          cross_point_sphere[0]) + myRay.get_radius_vector()
    finish_incident_sphere2 = np.multiply(myRay.get_direction_vector(),
                                          cross_point_sphere[1]) + myRay.get_radius_vector()
    print("cross_point_sphere_1: ", finish_incident_sphere1)
    print("cross_point_sphere_2: ", finish_incident_sphere2)

print("direction vector ",
      myRay.get_direction_vector())
print("law of reflection for surface: ",
      law_refl(myRay.get_direction_vector(), mySurf.get_normal()))
print("law of refraction for surface: ",
      law_refr(myRay.get_direction_vector(), mySurf.get_normal(), n1, n2))

focusing()


print("s: ", get_s(myRay.get_direction_vector(), mySurf.get_normal()))
print("p: ", get_p(myRay.get_direction_vector(), get_s(myRay.get_direction_vector(), mySurf.get_normal())))
t = [0, 0, 0]
t[0] = 1
t[1] = 1
t[2] = (myRay.get_direction_vector()[0] * t[0] +
        myRay.get_direction_vector()[1] * t[1]) * (-1) / myRay.get_direction_vector()[2]
length_t = (t[0] ** 2 + t[1] ** 2 + t[2] ** 2) ** 0.5
t = np.divide(t, length_t)
length_b = np.cross(myRay.get_direction_vector(),
                    t).dot(np.cross(myRay.get_direction_vector(), t)) ** 0.5
b = np.divide(np.cross(myRay.get_direction_vector(), t), length_b)
print('e ', myRay.get_direction_vector())
print('t ', t)
print('b ', b)

# Es = E_s(E1, t, get_s(myRay.get_direction_vector(), mySurf.get_normal()), E2, b)
# Ep = E_p(E1, t, E2, b, get_p(myRay.get_direction_vector(), get_s(myRay.get_direction_vector(), mySurf.get_normal())))

Es = E1 * get_s(myRay.get_direction_vector(), mySurf.get_normal()).dot(t) + \
     E2 * get_s(myRay.get_direction_vector(), mySurf.get_normal()).dot(b)
Ep = E1 * get_p(myRay.get_direction_vector(), get_s(myRay.get_direction_vector(), mySurf.get_normal())).dot(t) + E2 * \
     get_p(myRay.get_direction_vector(), get_s(myRay.get_direction_vector(), mySurf.get_normal())).dot(b)


print("Es: ", Es)
print("Ep: ", Ep)
cos_a = np.fabs(myRay.get_direction_vector().dot(mySurf.get_normal()))
cos_b = np.fabs(law_refr(myRay.get_direction_vector(), mySurf.get_normal(),
                         1, 1.5).dot(mySurf.get_normal()))
print("cos_a: ", cos_a)  # cos_a = 0.555
print("cos_b: ", cos_b)  # cos_b = 0.833
Ep_refract = Ep * (2 * n1 * cos_a) / (n2 * cos_a + n1 * cos_b)
Es_refract = Es * (2 * n1 * cos_a) / (n1 * cos_a + n2 * cos_b)
print("Es_refract: ", Es_refract)
print("Ep_refract: ", Ep_refract)
Ep_reflect = Ep * (n2 * cos_a - n1 * cos_b) / (n2 * cos_a + n1 * cos_b)
Es_reflect = Es * (n1 * cos_a - n2 * cos_b) / (n1 * cos_a + n2 * cos_b)
print("Es_reflect: ", Es_reflect)
print("Ep_reflect: ", Ep_reflect)
tangent_new = get_s(myRay.get_direction_vector(), mySurf.get_normal())
bitangent_new = get_p(law_refr(myRay.get_direction_vector(), mySurf.get_normal(), n1, n2), get_s(myRay.get_direction_vector(), mySurf.get_normal()))
print("tangent_new: ", tangent_new)
print("bitangent_new: ", bitangent_new)
print(n1 * (np.abs(Ep) ** 2 + np.abs(Es) ** 2) * cos_a)
print(n2 * (np.abs(Ep_refract) ** 2 + np.abs(Es_refract) ** 2) * cos_b + n1 *
      (np.abs(Ep_reflect) ** 2 + np.abs(Es_reflect) ** 2) * cos_a)


x, y, z = makeSurface(normal_surface, point_surface)
fig = plt.figure()
axes = fig.add_subplot(projection='3d')
axes.plot_surface(x, y, z)
axes.set_xlim([-10, 10])
axes.set_ylim([-10, 10])
axes.set_zlim([-10, 10])
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_zlabel('Z')
start_incident = myRay.get_radius_vector()
incident = myRay.get_direction_vector() * cross_point_surface
axes.quiver(start_incident[0], start_incident[1], start_incident[2], incident[0], incident[1], incident[2], color = 'r')
start_refr_refl = finish_incident_surface
normal = 5 * mySurf.get_normal()
axes.quiver(start_refr_refl[0], start_refr_refl[1], start_refr_refl[2], normal[0], normal[1], normal[2])
refl = 5 * law_refl(myRay.get_direction_vector(), mySurf.get_normal())
axes.quiver(start_refr_refl[0], start_refr_refl[1], start_refr_refl[2],
            refl[0], refl[1], refl[2], color='g')
refr = 5 * law_refr(myRay.get_direction_vector(), mySurf.get_normal(), n1,
                    n2)
axes.quiver(start_refr_refl[0], start_refr_refl[1], start_refr_refl[2],
            refr[0], refr[1], refr[2], color='y')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = mySh.get_radius_vector()[0] + mySh.get_radius() * np.outer(np.cos(u), np.sin(v))
y = mySh.get_radius_vector()[1] + mySh.get_radius() * np.outer(np.sin(u), np.sin(v))
z = mySh.get_radius_vector()[2] + mySh.get_radius() * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, rstride=1, cstride=1, color='y', alpha=0.7)

start_incident = myRay.get_radius_vector()
incident2 = myRay.get_direction_vector() * cross_point_sphere[0]
incident = myRay.get_direction_vector() * cross_point_sphere[1]
ax.quiver(start_incident[0], start_incident[1], start_incident[2], incident[0], incident[1], incident[2], color = 'r')
ax.quiver(start_incident[0] + incident[0], start_incident[1] + incident[1], start_incident[2] + incident[2], incident2[0] - incident[0], incident2[1] - incident[1], incident2[2] - incident[2], color = 'r')
print(mySh.get_normal_sphere(start_incident + incident))
start_refr_refl1 = finish_incident_sphere2
normal = 5 * mySh.get_normal_sphere(start_incident + incident)
ax.quiver(start_refr_refl1[0], start_refr_refl1[1], start_refr_refl1[2], normal[0], normal[1], normal[2])
refl = 5 * law_refl(myRay.get_direction_vector(),mySh.get_normal_sphere(start_incident + incident))
ax.quiver(start_refr_refl1[0], start_refr_refl1[1], start_refr_refl1[2],refl[0], refl[1], refl[2], color = 'g')
refr = 5 * law_refr(myRay.get_direction_vector(),mySh.get_normal_sphere(start_incident + incident), n1, n2)
ax.quiver(start_refr_refl1[0], start_refr_refl1[1], start_refr_refl1[2],refr[0], refr[1], refr[2], color = 'b')
plt.show()

teta_i = np.arccos(cos_a)
# draw_ellipse(Es, Ep)
# draw_ellipse(Es_refract, Ep_refract)
# draw_ellipse(Es_reflect, Ep_reflect)
plt.figure()
polar_ellipse(np.array([Es, Ep]), "")
plt.plot()
plt.show()
plt.figure()
polar_ellipse(np.array([Es_refract, Ep_refract]), "")
plt.plot()
plt.show()
plt.figure()
polar_ellipse(np.array([Es_reflect, Ep_reflect]), "")
plt.plot()
plt.show()
