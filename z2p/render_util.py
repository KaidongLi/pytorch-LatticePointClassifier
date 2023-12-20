import math

import torch

from . import data

world_mat_object = torch.tensor([
    [0.5085, 0.3226, 0.7984, 0.0000],
    [-0.3479, 0.9251, -0.1522, 0.0000],
    [-0.7877, -0.2003, 0.5826, 0.3384],
    [0.0000, 0.0000, 0.0000, 1.0000]
])

world_mat_inv = torch.tensor([
    [0.4019, 0.9157, 0.0000, 0.3359],
    [-0.1932, 0.0848, 0.9775, -1.0227],
    [0.8951, -0.3928, 0.2110, -7.0748],
    [-0.0000, 0.0000, -0.0000, 1.0000]
])

proj = torch.tensor([
    [2.1875, 0.0000, 0.0000, 0.0000],
    [0.0000, 3.8889, 0.0000, 0.0000],
    [0.0000, 0.0000, -1.0020, -0.2002],
    [0.0000, 0.0000, -1.0000, 0.0000]
])


def generate_roation(phi_x, phi_y, phi_z):
    def Rx(theta):
        return torch.tensor([[1, 0, 0],
                             [0, math.cos(theta), -math.sin(theta)],
                             [0, math.sin(theta), math.cos(theta)]])

    def Ry(theta):
        return torch.tensor([[math.cos(theta), 0, math.sin(theta)],
                             [0, 1, 0],
                             [-math.sin(theta), 0, math.cos(theta)]])

    def Rz(theta):
        return torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                             [math.sin(theta), math.cos(theta), 0],
                             [0, 0, 1]])

    return Rz(phi_z) @ Ry(phi_y) @ Rx(phi_x)


def rotate_pc(pc, rx, ry, rz):
    rotation = generate_roation(rx, ry, rz)
    rotated = pc.clone()
    rotated[:, :3] = rotated[:, :3] @ rotation.T
    if rotated.shape[-1] == 6:
        rotated[:, 3:] = rotated[:, 3:] @ rotation.T
    return rotated


def draw_pc(pc: torch.Tensor, res=(540, 960), radius=5, timer=None, dy=0, scale=1):
    xyz = pc[:, :3]
    xyz -= xyz.mean(dim=0)
    t_scale = xyz.norm(dim=-1).max()
    xyz /= t_scale
    xyz *= scale

    xyz[:, -1] += xyz[:, -1].min()

    n, _ = xyz.shape

    if timer is not None:
        with timer('project'):
            xyz_pad = torch.cat([xyz, torch.ones_like(pc[:, :1])], dim=-1)
            xyz_local = xyz_pad @ world_mat_inv.T
            distances = -xyz_local[:, 2]

            projected = xyz_local @ proj.T
            projected = projected / projected[:, 3:4]
            projected = projected[:, :3]

            u_pix = ((projected[0] + 1) / 2) * res[1]
            v_pix = ((projected[1] + 1) / 2) * res[0] + dy

        with timer('z-buffer'):
            z_buffer = data.scatter(u_pix, v_pix, distances, res, radius=radius)[:, :]
    else:
        xyz_pad = torch.cat([xyz, torch.ones_like(pc[:, :1])], dim=-1)
        xyz_local = xyz_pad @ world_mat_inv.T
        distances = -xyz_local[:, 2]

        projected = xyz_local @ proj.T
        projected = projected / projected[:, 3:4]
        projected = projected[:, :3]

        u_pix = ((projected[:, 0] + 1) / 2) * res[1]
        v_pix = ((projected[:, 1] + 1) / 2) * res[0] + dy

        z_buffer = data.scatter(u_pix.numpy(), v_pix.numpy(), distances, res, radius=radius)[:, :]

    z_buffer = z_buffer[data.RANGES[0][0]: data.RANGES[0][1], :]
    z_buffer = z_buffer[:, data.RANGES[1][0]:data.RANGES[1][1]]
    z_buffer = data.resize(z_buffer)
    return z_buffer


def generate_controls_vector(n_style=6):
    rgb = [255, 255, 255]
    light = [0, 0, 0]
    # create an empty style vector
    controls = torch.zeros(n_style).float()

    # RGB
    controls[0], controls[1], controls[2] = rgb[2], rgb[1], rgb[0]
    controls[:3] /= 255
    controls[0:3] = controls[0:3].clamp(0, 0.95)

    # Light position
    # delta_r, delta_phi, delta_theta
    controls[3], controls[4], controls[5] = light[0], light[1], light[2]

    # limit phi
    controls[4] = controls[4].clamp(-math.pi / 4, math.pi / 4)
    # limit theta
    controls[5] = controls[5].clamp(0, math.pi / 4)

    if n_style == 8:
        metal = 0.5
        roughness = 0.5
        # if in metal roughness mode set metal roughness values as well
        controls[6], controls[7] = metal, roughness
        # limit values between 0-1 i.e. the values the network was trained on
        controls[6] = controls[6].clamp(0, 1)
        controls[7] = controls[7].clamp(0, 1)

    return controls


def embed_color(img: torch.Tensor, color, box_size=70):
    shp = img.shape
    D2 = [shp[2] - box_size, shp[2]]
    D3 = [shp[3] - box_size, shp[3]]
    img = img.clone()
    img[:, :3, D2[0]:D2[1], D3[0]:D3[1]] = color[:, :, None, None]
    if img.shape[1] == 4:
        img[:, -1, D2[0]:D2[1], D3[0]:D3[1]] = 1
    return img