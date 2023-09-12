#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path

import random
# if torch.__version__ == '1.6.0+cu101' and sys.platform.startswith('linux'):
#     get_ipython().system('pip install pytorch3d')
# else:
#     need_pytorch3d = False
#     try:
#         import pytorch3d
#     except ModuleNotFoundError:
#         need_pytorch3d = True
#     if need_pytorch3d:
#         get_ipython().system('curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz')
#         get_ipython().system('tar xzf 1.10.0.tar.gz')
#         os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
#         get_ipython().system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")
from pytorch3d.io import load_obj, IO, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    DirectionalLights,
    TexturesAtlas,
    TexturesUV,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer
)
import matplotlib.pyplot as plt
import matplotlib
from utils import Params


# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Import parameters
params = Params("params_paper_zgt2.json")
params = Params("params_paper.json")
obj_filename = params.obj_filename


def get_mesh(obj_filename, device):
    """
    Generates Meshes object and initializes the mesh with vertices, faces,
    and textures.

    Args:
        obj_filename: str, path to the 3D obj filename
        device: str, the torch device containing a device type ('cpu' or
        'cuda')

    Returns:
        mesh: Meshes object
    """
    ext = os.path.splitext(obj_filename)[-1]

    if ext == '.obj' or ext == '.stl':
        # Get vertices, faces, and auxiliary information
        verts, faces, aux = load_obj(
            obj_filename,
            device=device,
            # load_textures=True,
            # create_texture_atlas=True,
            # texture_atlas_size=4,
            # texture_wrap="repeat"
            )

        # Create a textures object
        # atlas = aux.texture_atlas

        white_texture = torch.ones_like(verts)  # White color
        textures = TexturesVertex(verts_features=[white_texture])

        # Create Meshes object
        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=textures #TexturesAtlas(atlas=[atlas]),
        ) 
        
    elif ext == '.ply':
        verts, faces = load_ply(obj_filename)

        white_texture = torch.ones_like(verts)  # White color
        textures = TexturesVertex(verts_features=[white_texture])

        # Create a textures object
        mesh = Meshes(verts=[verts], 
                      faces=[faces],
                      textures=textures
                      ).to(device=device)

    else:
        print(f"ERROR: Unexpected Extension: {ext}")
        exit()

    
    return mesh

def init_rasterizer(image_size: int = 512, 
                    dist: float = 1.0, 
                    device: torch.device = torch.device('cpu'), 
                    elev: int = 180, 
                    azim: int = 0, 
                    znear: int = 1, 
                    zfar: int = 100, 
                    fov: int = 60):
    """
    Generates a mesh renderer by combining a rasterizer and a shader.

    Args:
        image_size: int, the size of the rendered .png image
        dist: int, distance between the camera and 3D object
        device: str, the torch device containing a device type ('cpu' or
        'cuda')
        elev: list, contains elevation values
        azim: list, contains azimuth angle values

    Returns:
        renderer: MeshRenderer class
    """
    # Initialize the camera with camera distance, elevation, azimuth angle,
    # and image size
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, zfar=zfar, fov=fov)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=znear, zfar=zfar)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    # Initialize rasterizer by using a MeshRasterizer class
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    return rasterizer

def get_renderer(rasterizer: MeshRasterizer,
                 device: torch.device = torch.device('cpu'), 
                 light_dir = None):
    """
    Generates a mesh renderer by combining a rasterizer and a shader.

    Args:
        image_size: int, the size of the rendered .png image
        dist: int, distance between the camera and 3D object
        device: str, the torch device containing a device type ('cpu' or
        'cuda')
        elev: list, contains elevation values
        azim: list, contains azimuth angle values

    Returns:
        renderer: MeshRenderer class
    """
    # Initialize the camera with camera distance, elevation, azimuth angle,
    # # and image size
    # R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, zfar=zfar, fov=fov)
    # raster_settings = RasterizationSettings(
    #     image_size=image_size,
    #     blur_radius=0.0,
    #     faces_per_pixel=1,
    # )
    # # Initialize rasterizer by using a MeshRasterizer class
    # rasterizer = MeshRasterizer(
    #     cameras=cameras,
    #     raster_settings=raster_settings
    # )

    lights = DirectionalLights(device=device, 
                               ambient_color=((0.1, 0.1, 0.1), ),
                               diffuse_color=((0.5, 0.5, 0.5), ),
                               specular_color=((0.2, 0.2, 0.2), ),
                               direction=(light_dir,)) if light_dir is not None else None
    
    # The textured phong shader interpolates the texture uv coordinates for
    # each vertex, and samples from a texture image.
    shader = SoftPhongShader(device=device, cameras=rasterizer.cameras, lights=lights)

    # Create a mesh renderer by composing a rasterizer and a shader
    renderer = MeshRenderer(rasterizer, shader)

    return renderer


def render_image(renderer, mesh, obj_filename, index):
    """
    Renders an image using MeshRenderer class and Meshes object. Saves the
    rendered image as a .png file.

    Args:
        image_size: int, the size of the rendered .png image
        dist: int, distance between the camera and 3D object
        device: str, the torch device containing a device type ('cpu' or
        'cuda')
        elev: list, contains elevation values
        azim: list, contains azimuth angle values

    Returns:
        renderer: MeshRenderer class
    """

    # def normalize_2d(matrix):
    #     norm = np.linalg.norm(matrix)
    #     matrix = matrix/norm  # normalized matrix
    #     return matrix

    image = renderer(mesh)
    # image = normalize_2d(image.cpu()).to(device)

    out = os.path.normpath(obj_filename).split(os.path.sep)
    mesh_filename = out[-1].split(".")[0]

    dir_to_save = f"out/{mesh_filename}"
    os.makedirs(dir_to_save, exist_ok=True)
    
    file_to_save = f'Img{index:03}.png'

    filename = os.path.join(dir_to_save, file_to_save)
    matplotlib.image.imsave(filename, image[0, ..., :3].cpu().numpy())
    print("Saved image as " + str(filename))
    

def main(): 
    random.seed(42)

    light_directions = [[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, -0.5)] for i in range(20)]
    light_directions = [[0, 0, -1]]

    # print(light_directions[0])
    for i, l in enumerate(light_directions):
        light_directions[i] = l / np.sqrt(np.sum([i ** 2 for i in l]))
    # print(light_directions[0])

    rasterizer = init_rasterizer(image_size=params.image_size, dist=params.camera_dist, device=device, 
                                 elev=params.elevation, azim=params.azim_angle, znear=params.znear, zfar=params.zfar,
                                 fov=params.fov)
    
    mesh = get_mesh(obj_filename, device)

    depth_map = rasterizer(mesh).zbuf.squeeze()
    print(f"Shape: {depth_map.shape}")
    matplotlib.image.imsave("test.png", depth_map.cpu().numpy())

    for i, light in enumerate(light_directions):
        renderer = get_renderer(rasterizer=rasterizer, device=device, light_dir=light)
        render_image(renderer=renderer, mesh=mesh, obj_filename=obj_filename, index=i)
     


if __name__ == "__main__":
    main()
