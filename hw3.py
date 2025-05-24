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
            sh=150, refl=0.05, refr=0.0):
        return dict(ambient=ambient, diffuse=diffuse,
                    specular=spec, shininess=sh,
                    reflection=refl, refraction=refr)

    # ─── materials ──────────────────────────────────────────────
    floor_mat   = mat((0.05,0.08,0.05), (0.0,0.40,0.0), spec=(0.15,0.15,0.15), sh=60)
    sky_mat     = mat((0.05,0.05,0.08), (0.6,0.8,1.0), spec=(0,0,0), sh=1)
    diamond_mat = mat((0,0,0), [0.05,0.35,0.30], spec=(0.9,0.9,0.9), sh=300, refl=0.07, refr=0.90)

    # ─── geometry: floor & sky ──────────────────────────────────
    floor = Plane([0,1,0], [0,-1.0,0]);   floor.set_material(**floor_mat)
    sky   = Plane([0,0,1], [0,0,-6.0]);   sky.set_material(**sky_mat)

    # ─── geometry: raw diamond vertices ────────────────────────
    base_v = np.array([
        [-0.8, -0.25, -3.0],   # A
        [-0.06, 0.15, -2.3],   # B
        [ 0.8,  0.05, -3.0],   # C
        [-0.16, 1.05, -3.0],   # D (top)
        [ 0.34,-0.95, -3.0],   # E (bottom)
    ])
    # ─── rotate 45° about Y so that edge (A–C) faces the camera ─
    theta = np.deg2rad(45)
    rotY = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [             0, 1,             0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    center = np.mean(base_v, axis=0)
    base_v = ((base_v - center) @ rotY.T) + center

    # ─── build the diamond and propagate its material ──────────
    cloak = Diamond(base_v)
    cloak.set_material(**diamond_mat)
    cloak.apply_materials_to_triangles()

    objects = [floor, sky, cloak]

    # ─── a brighter “sun” so you can actually see those two front faces ──
    sun = DirectionalLight(intensity=np.array([2.5,2.5,2.3]),
                           direction=np.array([-1,-1,-0.4]))
    fill = DirectionalLight(intensity=np.array([0.6,0.7,0.8]),
                            direction=np.array([ 1,-1,-0.2]))
    lights = [sun, fill]

    # ─── camera & ambient ───────────────────────────────────────
    camera  = np.array([0.0, 0.0, 1.5])

    return camera, lights, objects

