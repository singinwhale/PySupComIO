from typing import List, Dict

from PySupComIO.common_types import Bone, Transform


class Pose:
    """
    Represents a pose of the entire skeleton
    """
    bone_transforms: Dict[Bone, Transform]  # maps each bone to a new transform (no scale)


class Frame:
    time: float
    flags: int
    pose: Pose


InvalidBoneLinkError = Exception


class BoneLink:
    name: str
    index: int
    parent: 'BoneLink'
    parent_index: int


class Animation:
    """
    Represents a Supreme Commander Animation (.SCA)
    """
    duration: float
    bone_links: List[BoneLink]
    initial_pose: Pose
    frames: List[Frame]
