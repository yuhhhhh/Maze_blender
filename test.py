# -*- coding: utf-8 -*-
# @Time    : 2020/11/14 4:01 下午
# @Author  : yuhang
# @FileName: test.py
# @Software: PyCharm
# @Email   : yuhang.1109@qq.com

import bpy
import time

bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))

for i in range(10):
    time.sleep(0.1)
    bpy.data.objects["Sphere"].location = (2 * i, 2 * i, 0)
