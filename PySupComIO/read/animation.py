from pathlib import Path
from typing import BinaryIO, List
import io
import struct
import numpy as np
import logging

from PySupComIO.animation import Animation, Frame, BoneLink, InvalidBoneLinkError, Pose
from PySupComIO.read._common import read_null_terminated_string
from PySupComIO.common_types import Bone, Transform

logger = logging.getLogger(__name__)

FileError = Exception
FileVersionError = Exception


class ScaHeader:
    version: int
    frames_num: int
    duration: float
    bones_num: int
    bone_names_offset: int
    bone_links_offset: int
    anim_offset: int
    frame_size: int  # size of a stored frame in bytes


def _read_header(file_reader: io.BufferedReader) -> ScaHeader:
    # read unpacked_header
    unpacked_header = struct.unpack_from('4s2If5I', file_reader, 0)

    if unpacked_header[0] is not "ANIM":
        raise FileError(f"Could not find ANIM header at start of file. Is ({unpacked_header[0]})"
                        "Are you sure that you selected a valid SCA file?")
    header = ScaHeader()
    header.version = unpacked_header[1]
    header.frames_num = unpacked_header[2]
    header.duration = unpacked_header[3]
    header.bones_num = unpacked_header[4]
    header.bone_names_offset = unpacked_header[5]
    header.bone_links_offset = unpacked_header[6]
    header.anim_offset = unpacked_header[7]
    header.frame_size = unpacked_header[8]

    if header.version is not 5:
        raise FileVersionError("This code is written with the assumption "
                               f"that the file version is 5 but it is {header.version}")

    return header


def _read_bone_links(header: ScaHeader, file_reader: io.BufferedReader) -> List[BoneLink]:
    bone_links: List[BoneLink] = []
    file_reader.seek(header.bone_names_offset, io.SEEK_SET)
    for i in range(0, header.bones_num):
        bone_link = BoneLink()
        bone_link.name = read_null_terminated_string(file_reader.tell(), file_reader)
        bone_link.index = i
        bone_links += bone_link

    bone_link_format = 'i'
    bone_link_size = struct.calcsize(bone_link_format)
    for i in range(0, header.bones_num):
        bone_link_offset = header.bone_links_offset + i * bone_link_size
        bone_link_data = struct.unpack_from(bone_link_format, file_reader, bone_link_offset)
        bone_parent_index = bone_link_data[0]
        bone_links[i].parent_index = bone_parent_index

    for bone_link in bone_links:
        if bone_link.parent_index >= 0:
            try:
                bone_link.parent = bone_links[bone_link.parent_index]
            except IndexError:
                raise InvalidBoneLinkError(f"Tried link to an invalid BoneLink with index {bone_link.parent_index}")

    return bone_links


def _read_pose(pose_offset: int, header: ScaHeader, animation: Animation, file_reader: io.BufferedReader) -> Pose:
    pose_format = "3f4f" * len(animation.bone_links)
    assert (header.frame_size == struct.calcsize(pose_format) - 8, "Pose is not the expected size")
    pose = Pose()

    pose_data = struct.unpack_from(pose_format, file_reader, pose_offset)

    for i in range(0, int(len(animation.bone_links) / 7)):
        transform = Transform()
        position_start = 7 * i
        rotation_start = position_start + 3
        transform.position = pose_data[position_start:position_start + 2]
        transform.rotation = pose_data[rotation_start:rotation_start + 3]
        pose.bone_transforms += transform

    return pose


def _read_frames(header: ScaHeader, animation: Animation, file_reader: io.BufferedReader) -> List[Frame]:
    frame_header_format = 'fi'
    frame_header_size = struct.calcsize(frame_header_format)
    pose_size = header.frame_size - frame_header_size

    frames_start_offset = header.anim_offset + pose_size
    frames: List[Frame] = []
    for i in range(0, header.frames_num):
        frame_offset = frames_start_offset + i * header.frames_num
        frame_header_data = struct.unpack_from(frame_header_format, file_reader, frame_offset)
        frame = Frame()
        frame.time = frame_header_data[0]
        frame.flags = frame_header_data[1]
        frame.pose = _read_pose(frame_offset+frame_header_size, header, animation, file_reader)
        frames += frame

    return frames


def read_animation(filepath: Path) -> Animation:
    """
    Reads the SCA Animation at the given filepath
    :param filepath: path to a SCA Animation file
    :return: returns the animation that is in the given file
    """
    if filepath.is_dir():
        raise IsADirectoryError

    if not filepath.exists():
        raise FileNotFoundError(filepath)

    file_reader: io.BufferedReader
    with filepath.open('rb') as file_reader:
        animation = Animation()

        logger.info(f"Reading SCA Model from {filepath}")

        header = _read_header(file_reader)

        animation.duration = header.duration
        animation.bone_links = _read_bone_links(header, file_reader)
        animation.initial_pose = _read_pose(header.anim_offset, header, animation, file_reader)
        animation.frames = _read_frames(header, animation, file_reader)
        return animation
