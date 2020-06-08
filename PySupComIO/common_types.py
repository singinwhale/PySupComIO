from typing import (List)

import numpy as np


class Transform:
    """
    Represents a 3D Transform without scale
    """
    position: List[float]  # position vector x y z
    rotation: List[float]  # quaternion w x y z


class Bone:
    """
    Represents a bone in the bone hierarchy of the model
    """
    name: str = ""
    index: int = -1 # the index of this bone when it has to be referred to by index
    parent: 'Bone' = None
    parent_index: int = -1
    inverse_rest_pose_matrix: np.array = np.identity(3)
    transform: Transform = Transform()  # transform of this bone relative to its parent. Affine transform without scale

