'''
Meshing functions.
'''

# ======= Imports =======

import gmsh
import torch
import numpy as np

from torch import Tensor
from typing import (
    List,
    Tuple,
)

from typing import Any

# ======= Functions =======


def add_physical_fuild_marker(dim: int) -> List[Tuple[Any, Any]]:
    fluid_marker = 1
    volumes = gmsh.model.getEntities(dim=dim)

    if len(volumes) != 1:
        raise ValueError("There should be only 1 volume.")

    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

    return volumes


def tag_surfaces_to_meshes(
    volumes: List[Tuple[Any, Any]],
    length: float,
    height: float,
) -> None:
    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
    inflow, outflow, walls, obstacle = [], [], [], []

    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(
            boundary[0],
            boundary[1],
        )
        if np.allclose(center_of_mass, [0, height / 2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [length, height / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(
            center_of_mass,
            [length / 2, height, 0]
        ) or np.allclose(center_of_mass, [length / height, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")


def generate_mesh(gdim: int) -> None:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")


def set_variable_mesh_sizes(
    obstacle: Any,
    radius: float,
    height: float,
) -> None:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", [obstacle])
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", radius / 20)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.05 * height)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", radius)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * height)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(
        min_field, "FieldsList", [threshold_field]
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)


def gmsh_to_tensor() -> Tensor:
    _, node_coords, _ = gmsh.model.mesh.getNodes()

    # Take only x, y
    coords = np.array(node_coords).reshape(-1, 3)[:, :2]

    return torch.tensor(coords, dtype=torch.float32)
