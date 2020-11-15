import bpy
import numpy as np

maze = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
)


for i in range(8):
    for j in range(8):
        if maze[i, j] == 1:
            bpy.ops.mesh.primitive_cube_add(location = (2*i, 2*j, 0))