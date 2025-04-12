import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
    label as ndi_label
)
import trimesh
from skimage.measure import marching_cubes
from skimage.draw import line_nd
from scipy.spatial import cKDTree
from typing import List, Dict, Optional, Tuple, Set, Union, Any


class VolumeConnector:
    """
    Connects disconnected binary components in a 3D volume via minimal paths.

    This class identifies separate connected components in a binary 3D array
    and iteratively links them by finding the shortest voxel-to-voxel paths 
    on their surfaces. It updates the volume to form a single connected object.
    """

    def __init__(self, volume: np.ndarray) -> None:
        if not isinstance(volume, np.ndarray):
            raise TypeError("Input volume must be a numpy array.")
        if volume.ndim != 3:
            raise ValueError("Input volume must be a 3D numpy array.")
        if not np.issubdtype(volume.dtype, np.bool_):
            volume = volume.astype(bool)
        self.volume: np.ndarray = volume
        self.labeled_volume: Optional[np.ndarray] = None
        self.surface_voxels: Dict[int, np.ndarray] = {}
        self.component_labels: np.ndarray = np.array([])
        self.fixed_volume: Optional[np.ndarray] = None

    def find_connected_components(self) -> None:
        """
        Label connected components in the volume using a standard 3D
        structuring element.
        """
        structure = generate_binary_structure(3, 1)
        self.labeled_volume, num_features = ndi_label(
            self.volume, structure=structure)
        self.component_labels = np.unique(self.labeled_volume)
        self.component_labels = self.component_labels[self.component_labels != 0]

    def get_surface_voxels(self) -> None:
        """
        Identify surface voxels of each component by binary erosion.
        """
        self.surface_voxels = {}
        structure = generate_binary_structure(3, 1)
        for label_id in self.component_labels:
            component = (self.labeled_volume == label_id)
            eroded = binary_erosion(component, structure=structure,
                                    border_value=0)
            surface = component & ~eroded
            coords = np.array(np.nonzero(surface)).T
            self.surface_voxels[label_id] = coords

    def get_largest_component_label(self) -> int:
        """
        Find the label of the largest connected component by voxel count.
        """
        sizes = {}
        for label_id in self.component_labels:
            sizes[label_id] = np.sum(self.labeled_volume == label_id)
        largest_label = max(sizes, key=sizes.get)
        return largest_label

    def interpolate_line(self, point1: Tuple[int, int, int],
                         point2: Tuple[int, int, int]) -> np.ndarray:
        """
        Compute voxel indices along a line segment defined by two points.
        """
        return np.array(line_nd(point1, point2)).T

    def connect_components(self) -> None:
        """
        Link all disconnected components by adding binary lines between 
        closest surface voxels, iteratively forming a single connected object.
        """
        self.fixed_volume = self.volume.copy()
        largest_label = self.get_largest_component_label()
        connected_labels = {largest_label}
        unconnected_labels = set(self.component_labels) - connected_labels
        kdtrees = {}
        for label_id in self.component_labels:
            kdtrees[label_id] = cKDTree(self.surface_voxels[label_id])
        while unconnected_labels:
            min_dist = np.inf
            min_labels = (None, None)
            min_voxel_pair = (None, None)
            for label_u in unconnected_labels:
                coords_u = self.surface_voxels[label_u]
                for label_c in connected_labels:
                    distances, indices = kdtrees[label_c].query(
                        coords_u, k=1)
                    idx = np.argmin(distances)
                    dist = distances[idx]
                    if dist < min_dist:
                        min_dist = dist
                        min_labels = (label_u, label_c)
                        voxel_u = coords_u[idx]
                        voxel_c = self.surface_voxels[label_c][indices[idx]]
                        min_voxel_pair = (voxel_u, voxel_c)
            label_u, label_c = min_labels
            voxel_u, voxel_c = min_voxel_pair
            line_coords = self.interpolate_line(voxel_u, voxel_c)
            self.fixed_volume[tuple(line_coords.T)] = True
            structure = generate_binary_structure(3, 1)
            line_volume = np.zeros_like(self.volume, dtype=bool)
            line_volume[tuple(line_coords.T)] = True
            for _ in range(5):
                line_volume = binary_dilation(line_volume,
                                              structure=structure)
            self.fixed_volume = self.fixed_volume | line_volume
            connected_labels.add(label_u)
            unconnected_labels.remove(label_u)
        self.labeled_volume, _ = ndi_label(
            self.fixed_volume, structure=generate_binary_structure(3, 1))
        self.component_labels = np.unique(self.labeled_volume)
        self.component_labels = self.component_labels[self.component_labels != 0]

    def fix_volume(self) -> np.ndarray:
        """
        Execute the pipeline to connect all components in the input volume.

        Returns
        -------
        ndarray
            Connected binary 3D volume.
        """
        self.find_connected_components()
        self.get_surface_voxels()
        self.connect_components()
        return self.fixed_volume


class TestVolumeConnector(VolumeConnector):
    """
    Test class for VolumeConnector using a synthetic volume.
    """

    def __init__(self: "TestVolumeConnector") -> None:
        volume = self.construct_test_volume()
        super().__init__(volume)

    def construct_test_volume(self: "TestVolumeConnector") -> np.ndarray:
        """
        Construct a synthetic 3D binary volume with four disconnected cubes.
        """
        volume = np.zeros((100, 100, 100), dtype=bool)
        volume[10:30, 10:30, 10:30] = True
        volume[70:90, 10:30, 10:30] = True
        volume[10:30, 70:90, 10:30] = True
        volume[10:30, 10:30, 70:90] = True
        return volume

    def run_test(self: "TestVolumeConnector") -> None:
        """
        Connect the components and verify the result merges into a single component.
        """
        fixed_volume = self.fix_volume()
        structure = generate_binary_structure(3, 1)
        _, num_components_before = ndi_label(
            self.volume, structure=structure
        )
        _, num_components_after = ndi_label(
            fixed_volume, structure=structure
        )
        results = []
        results.append(
            "Number of connected components before fixing: " +
            str(num_components_before)
        )
        results.append(
            "Number of connected components after fixing: " +
            str(num_components_after)
        )
        test_passed = num_components_after == 1
        results.append("Test passed." if test_passed else "Test failed.")
        print(results[0])
        print(results[1])
        print(results[2])
        with open("test_results.txt", "w") as f:
            f.write("\n".join(results))
        assert test_passed, "Test failed."

    def visualize(self: "TestVolumeConnector") -> None:
        """
        Visualize the components before and after connecting.
        """
        verts_before, faces_before, _, _ = marching_cubes(
            self.volume.astype(float), level=0.5
        )
        mesh_before = trimesh.Trimesh(
            vertices=verts_before, faces=faces_before
        )
        mesh_before.export("before.stl")
        mesh_before.show()
        verts_after, faces_after, _, _ = marching_cubes(
            self.fixed_volume.astype(float), level=0.5
        )
        mesh_after = trimesh.Trimesh(
            vertices=verts_after, faces=faces_after
        )
        mesh_after.export("after.stl")
        mesh_after.show()