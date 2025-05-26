import numpy as np

EPSILON = 1e-6
# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, normal):
    v = vector - 2 * np.dot(vector, normal) * normal
    return normalize(v)


## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = normalize(direction)

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection_point):
        return Ray(intersection_point, -self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, P):
        return np.linalg.norm(self.position - P)

    # This function returns the light intensity at a point
    def get_intensity(self, P):
        # calculate distance between light source and intersection 
        # calculate and return the light intensity based on kc, kl, kq
        distance = self.get_distance_from_light(P)
        return self.intensity * (1 / (self.kq * (distance ** 2) + self.kl * distance + self.kc))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = np.array(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq
        # TODO

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(self.position - intersection)

    def get_intensity(self, intersection):
        v_d = normalize(self.direction)
        v = normalize(intersection - self.position)
        cos = np.dot(v, v_d)
        if cos < 0:
            return np.zeros(3)
        distance = self.get_distance_from_light(intersection)
        return self.intensity * (np.dot(v, v_d) / (self.kq * (distance ** 2) + self.kl * distance + self.kc))



class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf

        nearest_intersections = [
            hit
            for obj in objects
            if (hit := obj.intersect(self)) is not None
        ]
        if nearest_intersections:
            nearest_intersection = min(nearest_intersections, key=lambda x: x[0])
        else:
            return None

        if nearest_intersection:
            nearest_object = nearest_intersection[1]
            min_distance = nearest_intersection[0]

        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection, emission=np.array([0,0,0]), refraction=0):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        self.emission = emission
        self.refraction = refraction
        self.ambient    = np.array(ambient,   dtype=float)
        self.diffuse    = np.array(diffuse,   dtype=float)
        self.specular   = np.array(specular,  dtype=float)
        self.shininess  = float(shininess)
        self.reflection = float(reflection)
        self.emission   = np.array(emission,  dtype=float)
        self.refraction = float(refraction)

class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + EPSILON)
        if t > 0:
            return t, self
        else:
            return None

    def get_normal(self, P_hit):
        return self.normal


class Triangle(Object3D):
    # """
    #     C
    #     /\
    #    /  \
    # A /____\ B
    #
    # The fornt face of the triangle is A -> B -> C.
    #
    # """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self):
        return normalize(np.cross(self.b - self.a, self.c - self.a))

    def intersect(self, ray: Ray):
        origin, dir = ray.origin, ray.direction
        e1 = self.b - self.a
        e2 = self.c - self.a
        A = np.column_stack((-dir, e1, e2))
        S = origin - self.a
        try:
            t, u, v = np.linalg.solve(A, S)
        except np.linalg.LinAlgError:
            return None

        if t > EPSILON and u >= 0 and v >= 0 and u + v <= 1:
            return t, self  # distance along the ray

        return None

    def get_normal(self, P_hit):
        return self.normal


class Diamond(Object3D):
#     """
#             D
#             /\*\
#            /==\**\
#          /======\***\
#        /==========\***\
#      /==============\****\
#    /==================\*****\
# A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
#    \==================/****/
#      \==============/****/
#        \==========/****/
#          \======/***/
#            \==/**/
#             \/*/
#              E
#
#     Similar to Traingle, every from face of the diamond's faces are:
#         A -> B -> D
#         B -> C -> D
#         A -> C -> B
#         E -> B -> A
#         E -> C -> B
#         C -> E -> A
#     """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                 [4,1,0],
                 [4,2,1],
                 [2,4,0]]
        l = [Triangle(self.v_list[t[0]], self.v_list[t[1]], self.v_list[t[2]]) for t in t_idx]
        return l

    def apply_materials_to_triangles(self):
        for triangle in self.triangle_list:
            triangle.set_material(ambient=self.ambient, diffuse=self.diffuse, specular=self.specular, shininess=self.shininess, reflection=self.reflection)

    def intersect(self, ray: Ray):
        nearest_intersection = ray.nearest_intersected_object(self.triangle_list)
        if nearest_intersection:
            return nearest_intersection[1], nearest_intersection[0]

        return None


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        origin, direction = ray.origin, ray.direction
        L = origin - self.center
        a = 1
        b = 2.0 * np.dot(direction, L)
        c = np.dot(L, L) - self.radius ** 2
        delta = b*b - 4*c
        if delta < 0:
            return None
        sqrt_delta = np.sqrt(delta)
        t1 = (-b - sqrt_delta) / 2.0
        t2 = (-b + sqrt_delta) / 2.0
        t_hit = None

        if t1 > EPSILON:
            t_hit = t1
        elif t2 > EPSILON:
            t_hit = t2

        if t_hit is None:
            return None    # intersections are behind the origin

        return t_hit, self

    def get_normal(self, P_hit):
        return normalize(P_hit - self.center)
