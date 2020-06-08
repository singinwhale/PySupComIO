from pathlib import Path
from typing import BinaryIO, List
import io
import struct
import numpy as np
import logging

from PySupComIO.common_types import Bone
from PySupComIO.read._common import read_null_terminated_string
from PySupComIO.model import Model, Vertex, Triangle, IncompleteTriangle

logger = logging.getLogger(__name__)

FileError = Exception
FileVersionError = Exception


class ScmHeader:
    version: int
    bone_data_offset: int
    bone_count: int
    vertices_offset: int
    extra_vertices_offset: int
    num_vertices: int
    triangles_offset: int
    num_triangle_indexes: int
    info_string_offset: int
    info_string_length: int


def _read_bones(header: ScmHeader, file_reader: io.BufferedReader):
    bones: List[Bone] = []

    file_reader.seek(header.bone_data_offset, io.SEEK_SET)

    # read bone data
    bone_data_startoffset = file_reader.tell()
    bone_data_format = '16f3f4f4i'
    bone_data_size = struct.calcsize(bone_data_format)
    for i in range(0, header.bone_count - 1):
        offset = bone_data_startoffset + bone_data_size * i
        bone_data = struct.unpack_from(bone_data_format, file_reader, offset)
        bone = Bone()
        # SCM bones contain the inverse rest pose of the joint as affine matrix4x4.
        # That is the inverse transform of the bone relative to the local origin of the mesh (0,0,0)
        # with row major (i.e. D3D default ordering)
        bone.inverse_rest_pose_matrix = np.array([bone_data[0:3],
                                                  bone_data[4:7],
                                                  bone_data[8:11],
                                                  bone_data[12:15]
                                                  ])
        bone.transform.position = bone_data[16:18]
        bone.transform.rotation = bone_data[19:22]
        bone_name_offset = bone_data[23]
        bone.parent_index = bone_data[24]

        bone.name = read_null_terminated_string(bone_name_offset, file_reader)
        bone.index = i

        bones += bone
        logger.info(f"Read Bone called '{bone.name}'")

    for bone in bones:
        bone.parent = bones[bone.parent_index]

    return bones


def _read_vertices(header: ScmHeader, bones: List[Bone], file_reader: io.BufferedReader):
    file_reader.seek(header.vertices_offset)
    vertices = []
    vertex_format = '3f3f3f3f2f2f4B'
    vertex_data_size = struct.calcsize(vertex_format)
    for i in range(0, header.num_vertices):
        vertex_offset = header.vertices_offset + i * vertex_data_size
        vertex_data = struct.unpack_from(vertex_format, file_reader, vertex_offset)
        vertex = Vertex()
        vertex.index = i
        vertex.position = vertex_data[0:2]
        vertex.normal = vertex_data[3:5]
        vertex.tangent = vertex_data[6:8]
        vertex.binormal = vertex_data[9:11]
        vertex.UVs = [vertex_data[12:13],
                      vertex_data[14:15]]

        # try associate the bones with the vertices
        for parent_bone_field_index in range(16, 19):
            bone_index = vertex_data[parent_bone_field_index]
            if bone_index == 255:
                continue

            try:
                vertex.bones += bones[bone_index]
            except IndexError:
                continue

    logger.info(f"Read {len(vertices)} vertices")
    return vertices


def _read_triangles(header: ScmHeader, vertices: List[Vertex], file_reader: io.BufferedReader) -> List[Triangle]:
    assert (ScmHeader.num_triangle_indexes % 3 == 0, "There are more vertex indexes than there should be "
                                                     "if the mesh would only contain triangles")

    file_reader.seek(header.triangles_offset)
    triangles = []
    triangle_format = '3H'
    triangle_size = struct.calcsize(triangle_format)
    for i in range(0, int(header.num_triangle_indexes / 3)):
        try:
            triangle_offset = header.triangles_offset + i * triangle_size
            triangle_data = struct.unpack_from(triangle_format, file_reader, triangle_offset)
            triangle = Triangle()
            triangle.vertex_indexes = triangle_data

            # create references to the vertices
            for vertex_index in triangle.vertex_indexes:
                triangle.vertices += vertices[vertex_index]
                assert vertices[vertex_index].index == vertex_index, "Vertex index does not match its referenced index"
        except IndexError as error:
            raise IncompleteTriangle(f"Triangle {i} references missing vertices")

    return triangles


def _read_info(header: ScmHeader, file_reader: io.BufferedReader) -> str:
    file_reader.seek(header.info_string_offset)

    return file_reader.read(header.info_string_length).decode('ascii')


def _read_header(file_reader: io.BufferedReader) -> ScmHeader:
    # read unpacked_header
    unpacked_header = struct.unpack_from('4s11I', file_reader, 0)

    if unpacked_header[0] is not "MODL":
        raise FileError("Could not find MODL header at start of file."
                        "Are you sure that you selected a valid SCM file?")
    header = ScmHeader()
    # noinspection DuplicatedCode
    header.version = unpacked_header[1]
    header.bone_data_offset = unpacked_header[2]
    header.bone_count = unpacked_header[3]
    header.vertices_offset = unpacked_header[4]
    header.extra_vertices_offset = unpacked_header[5]
    header.num_vertices = unpacked_header[6]
    header.triangles_offset = unpacked_header[7]
    header.num_triangle_indexes = unpacked_header[8]  # this is 3*triangle_count
    header.info_string_offset = unpacked_header[9]
    header.info_string_length = unpacked_header[10]

    if header.version is not 5:
        raise FileVersionError("This code is written with the assumption that the file version is 5.")

    return header


def read_model(filepath: Path) -> Model:
    """
    Reads the SCM Model specified by filepath
    :param filepath: path to the SCM file
    :return: model data that is in the file
    """
    file_reader: io.BufferedReader
    with filepath.open('rb') as file_reader:
        model = Model()

        logger.info(f"Reading SCM Model from {filepath}")

        header = _read_header(file_reader)

        model.bones = _read_bones(header, file_reader)
        model.vertices = _read_vertices(header, model.bones, file_reader)
        model.faces = _read_triangles(header, model.vertices, file_reader)
        model.info = _read_info(header, file_reader)
        return model
