from typing import List

from PySupComIO.common_types import Bone


class Vertex:
    """
    Represents one vertex in the model. A vertex has up to 2 UV channels.
    It stores position as well as normal, binormal and tangent vectors.
    """
    index: int  # index of this vertex when it should be referenced via index
    position: List[float]
    normal: List[float]
    tangent: List[float]
    binormal: List[float]
    UVs: List[List[float]]  # can contain up to two uv coordinates as XY positions
    bones: List[Bone]  # up to 4 bones which this vertex is parented to


IncompleteTriangle = Exception


class Triangle:
    """
    Represents a single face in the mesh
    Note: smooth shaded faces have shared vertices with their neighbouring faces
        and the vertex normals point to the average up vector of the adjacent faces
        Flat shaded faces have split vertices/ normals
    """
    vertex_indexes: List[int]  # exactly three vertex indexes of the vertices that make up this face
    vertices: List[Vertex]


class Model:
    """
    Represents a Supreme Commander 3D Model (.SCM)
    """
    name: str
    bones: List[Bone]
    vertices: List[Vertex]
    faces: List[Triangle]
    info: str
