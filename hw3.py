import numpy as np

from helper_classes import *
import matplotlib.pyplot as plt

EPSILON = 1e-10


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            # This is the main loop where each pixel color is computed.
            intersection = ray.nearest_intersected_object(objects)

            color = get_color(ambient, lights, objects, max_depth, ray, intersection, 1) if intersection else np.zeros(3)
            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


def get_color(ambient, lights, objects, max_level, ray, intersection, level):
    hit_obj, t = intersection
    P_hit = ray.origin + t * ray.direction

    normal = hit_obj.normal

    P_hit += normal * EPSILON

    color = hit_obj.emission + hit_obj.ambient * ambient

    for light in lights:
        # get a ray toward the light
        Lray = light.get_light_ray(P_hit)
        Ldir = Lray.direction

        if in_shadow(P_hit, Ldir, objects, light):
            continue

        # diffuse = k_d * (N·L) * light.intensity
        diffuse = np.array(hit_obj.diffuse, dtype=float)
        light_col = np.array(light.get_intensity(P_hit), dtype=float)
        diff = diffuse * max(normal.dot(Ldir), 0.0) * light_col
        # specular = k_s * (R·V)^shininess * light.color
        V = -ray.direction
        R = reflected(-Ldir, normal)   # reflect L about N
        specular = np.array(hit_obj.specular, dtype=float)
        spec = specular * (max(np.dot(V, R), 0.0) ** hit_obj.shininess) * light_col
        color += diff + spec

    # 3) if we’ve hit recursion limit, return
    level += 1
    if level > max_level:
        return color


    if hit_obj.reflection > 0:
        refl_dir = reflected(ray.direction, normal)
        refl_ray = Ray(P_hit + 1e-5 * refl_dir, refl_dir)
        refl_inter = refl_ray.nearest_intersected_object(objects)
        if refl_inter:
            color += hit_obj.reflection * get_color(ambient, lights, objects,
                                              max_level, refl_ray, refl_inter, level)

    # if hit_obj.refraction > 0:
    #     ior_outside = 1.0               # assume air outside
    #     ior_inside  = getattr(hit_obj, "ior", 1.5)
    #     entering    = np.dot(ray.direction, normal) < 0
    #     ior_from, ior_to = (ior_outside, ior_inside) if entering else (ior_inside, ior_outside)
    #     N_use = normal if entering else -normal
    #
    #     refr_dir = refracted(ray.direction, N_use, ior_from, ior_to)
    #     if refr_dir is not None:
    #         refr_ray = Ray(P_hit, refr_dir)
    #         refr_hit = refr_ray.nearest_intersected_object(objects)
    #         if refr_hit:
    #             color += hit_obj.refraction * get_color(
    #                 ambient, lights, objects, max_level, refr_ray, refr_hit, level
    #             )


    return color


def refracted(I, N, ior_from, ior_to):
    """
    Compute refracted direction (returns None if total-internal reflection).
    I  : incident *normalized* direction (pointing *into* the surface)
    N  : outward *normalized* surface normal
    ior_from: index of refraction of incident medium
    ior_to  : index of refraction of transmitted medium
    """
    cosi = -np.dot(I, N)
    eta = ior_from / ior_to
    k = 1.0 - eta**2 * (1.0 - cosi**2)
    if k < 0.0:        # total internal reflection
        return None
    return eta * I + (eta * cosi - np.sqrt(k)) * N


def in_shadow(P, Ldir, objects, light, max_dist=np.inf):
    shadow_ray = Ray(P + Ldir, Ldir)
    hit = shadow_ray.nearest_intersected_object(objects)
    if not hit:
        return False

    obj, t = hit
    if hasattr(light, "position"):           # point light
        d_light = np.linalg.norm(light.position - P)
        return t < d_light
    else:                                    # directional light
        return True


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
