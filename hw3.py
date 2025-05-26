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
        inside_origin = P_hit - normal * EPSILON
        trans_dir = ray.direction

        trans_ray = Ray(inside_origin, trans_dir)
        trans_hit = trans_ray.nearest_intersected_object(objects)
        if trans_hit:
            color += hit_obj.refraction * get_color(
                ambient, lights, objects, max_level,
                trans_ray, trans_hit, level,
            )

    return color


def in_shadow(P, Ldir, objects, light, max_dist=np.inf):
    shadow_ray = Ray(P + EPSILON * Ldir, Ldir)
    hit = shadow_ray.nearest_intersected_object(objects)

    if not hit:
        return False

    obj, t = hit

    if obj.refraction > 0.0:
        return False

    if hasattr(light, "position"):  # point light
        d_light = np.linalg.norm(light.position - P)
        return t < d_light
    else:  # directional light
        return True


# Write your own objects and lights
# TODO
def your_own_scene():

    # ─── helper to build materials ──────────────────────────────
    def mat(ambient, diffuse, spec=(0.3,0.3,0.3),
            sh=150, refl=0.15, refr=0.0):
        return dict(ambient=ambient, diffuse=diffuse,
                    specular=spec, shininess=sh,
                    reflection=refl, refraction=refr)

    # ─── materials ──────────────────────────────────────────────
    floor_mat   = mat((0.05,0.08,0.05), (0.0,0.40,0.0), spec=(0.15,0.15,0.15), sh=60)
    sky_mat     = mat((0.05,0.05,0.08), (0.6,0.8,1.0), spec=(0,0,0), sh=1)

    # ─── geometry: floor & sky ──────────────────────────────────
    floor = Plane([0,1,0], [0,-0.8,0]);   floor.set_material(**floor_mat)
    sky   = Plane([0,0,1], [0,0,-6.0]);   sky.set_material(**sky_mat)
    # ─── pyramid vertices ────────────────────────────────────────
    P   = np.array([ 0.0,  2.0, -4.0])  # apex
    FTL = np.array([-2.0,  0.0, -2.0])  # front–left
    FTR = np.array([ 2.0,  0.0, -2.0])  # front–right
    BTR = np.array([ 2.0,  0.0, -6.0])  # back–right
    BTL = np.array([-2.0,  0.0, -6.0])  # back–left

    # ─── four triangular faces ───────────────────────────────────
    faces = [
        Triangle(P, FTL, FTR),  # front face
        Triangle(P, FTR, BTR),  # right face
        Triangle(P, BTR, BTL),  # back face
        Triangle(P, BTL, FTL),  # left face
    ]
    for tri in faces:
        tri.set_material(**mat((0.03, 0.04, 0.02), (0.0, 0.40, 0.0), refr=0.2))

    faces[0].set_material(**mat((0.01, 0.01, 0.01), (0.3, 0.20, 0.0), refr=0.4))
    faces[1].set_material(**mat((0.04, 0.02, 0.03), (0.2, 0.40, 0.0), refr=0.2))
    faces[1].set_material(**mat((0.01, 0.02, 0.04), (0.5, 0.20, 0.1), refr=0.2))

    inner = Sphere([0.0, 0.8, -4.0], 0.6)
    inner.set_material(
        ambient   = [0.0, 0.0, 0.0],
        diffuse   = [0.5, 0.1, 0.05],   # bright red
        specular  = [0.2, 0.2, 0.2],
        shininess = 100,
        reflection= 0.3,
        refraction= 0.0
    )

    objects = [floor, sky, inner] + faces

    # ─── a brighter “sun” so you can actually see those two front faces ──
    sun = DirectionalLight(intensity=np.array([0.5,2.5,2.3]),
                           direction=np.array([-0.5,-1,-0.5]))
    lights = [sun]

    # ─── camera & ambient ───────────────────────────────────────
    camera  = np.array([0.0, 0.5, 1.5])

    return camera, lights, objects

