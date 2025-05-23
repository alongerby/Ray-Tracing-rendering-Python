import numpy as np
import logging
from helper_classes import *
import matplotlib.pyplot as plt

EPSILON = 1e-4


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

            color = get_color(ambient, lights, objects, max_depth, ray, intersection, 1) if intersection else np.zeros(
                3)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def get_color(ambient, lights, objects, max_level, ray, intersection, level):
    hit_obj, t = intersection
    P_hit = ray.origin + t * ray.direction

    normal = hit_obj.get_normal(P_hit)

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
        R = reflected(-Ldir, normal)  # reflect L about N
        specular = np.array(hit_obj.specular, dtype=float)
        spec = specular * (max(np.dot(V, R), 0.0) ** float(hit_obj.shininess)) * light_col
        color += diff + spec

    # 3) if we’ve hit recursion limit, return
    level += 1
    if level > max_level:
        return color

    if np.dot(ray.direction, normal) > 0:
        normal = -normal

    if hit_obj.reflection > 0.0:
        refl_dir = reflected(ray.direction, normal)
        refl_ray = Ray(P_hit, refl_dir)
        refl_inter = refl_ray.nearest_intersected_object(objects)
        if refl_inter:
            color += hit_obj.reflection * get_color(ambient, lights, objects,
                                                    max_level, refl_ray, refl_inter, level)

    if hit_obj.refraction > 0:  # 1. material lets light through
        # 2. decide which way the ray is travelling
        outside_ior = 1.0  # assume air outside
        inside_ior = getattr(hit_obj, "refraction", 1.0)  # glass, water, etc.

        front_face = np.dot(ray.direction, normal) < 0  # entering the object?
        n_from, n_to = (outside_ior, inside_ior) if front_face else (inside_ior, outside_ior)
        N_use = normal if front_face else -normal  # flip normal when exiting

        # 3. Snell-law direction
        trans_dir = refracted(ray.direction, N_use, n_from, n_to)

        # 4. if no TIR, trace the transmitted ray recursively
        if trans_dir is not None:
            trans_ray = Ray(P_hit + EPSILON * trans_dir, trans_dir)
            trans_hit = trans_ray.nearest_intersected_object(objects)
            if trans_hit:
                color += hit_obj.refraction * get_color(
                    ambient, lights, objects, max_level,
                    trans_ray, trans_hit, level,
                )

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
    k = 1.0 - eta ** 2 * (1.0 - cosi ** 2)
    if k < 0.0:  # total internal reflection
        return None
    return eta * I + (eta * cosi - np.sqrt(k)) * N


def in_shadow(P, Ldir, objects, light, max_dist=np.inf):
    shadow_ray = Ray(P + EPSILON * Ldir, Ldir)
    hit = shadow_ray.nearest_intersected_object(objects)
    if not hit:
        return False

    obj, t = hit
    if hasattr(light, "position"):  # point light
        d_light = np.linalg.norm(light.position - P)
        return t < d_light
    else:  # directional light
        return True


# Write your own objects and lights
# TODO
def your_own_scene():
    # === geometry ===

    # floor + back wall (as before)
    floor = Plane([0, 1, 0], [0, -0.5, 0])
    floor.set_material(
        ambient   = [0.1, 0.1, 0.1],
        diffuse   = [0.8, 0.8, 0.8],
        specular  = [0.5, 0.5, 0.5],
        shininess = 300,
        reflection= 0.25,
        refraction= 0.0
    )

    background = Plane([0, 0, 1], [0, 0, -4])
    background.set_material(
        ambient   = [0.2, 0.2, 0.2],
        diffuse   = [0.25, 0.3, 0.5],
        specular  = [0.1, 0.1, 0.1],
        shininess = 20,
        reflection= 0.0,
        refraction= 0.0
    )

    # six glass‐box faces
    size = 1.0  # half‐extents
    glass_kwargs = {
        "ambient":   [0.0, 0.0, 0.0],
        "diffuse":   [0.0, 0.0, 0.0],
        "specular":  [0.9, 0.9, 1.0],
        "shininess": 200,
        "reflection":0.1,
        "refraction":0.9
    }

    # +X face
    right = Plane([-1, 0, 0], [ size, 0, 0])
    right.set_material(**glass_kwargs)
    # -X face
    left  = Plane([ 1, 0, 0], [-size, 0, 0])
    left.set_material(**glass_kwargs)
    # +Y face (ceiling)
    top   = Plane([0, -1, 0], [0,  size, 0])
    top.set_material(**glass_kwargs)
    # -Y face (floor already exists, but we can add a slightly lower glass cap)
    bottom= Plane([0,  1, 0], [0, -size, 0])
    bottom.set_material(**glass_kwargs)
    # +Z face

    objects = [
        floor, background,
        right, left, top, bottom, 
    ]

    # === two lights ===
    key = PointLight(
        intensity = np.array([1, 1, 1]),
        position  = np.array([2, 3, 2]),
        kc=0.1, kl=0.05, kq=0.02
    )
    fill = DirectionalLight(
        intensity = np.array([0.4, 0.45, 0.5]),
        direction = np.array([-1, -1, -0.5])
    )
    lights = [key, fill]

    # === camera & render ===
    camera  = np.array([0.0, 0.2, 1.5])
    ambient = np.array([0.05, 0.05, 0.05])

    return camera, lights, objects
