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
    name: str
    index: int  # the index of this bone when it has to be referred to by index
    parent: 'Bone'
    parent_index: int
    inverse_rest_pose_matrix: np.array
    transform: Transform  # transform of this bone relative to its parent. Affine transform without scale

