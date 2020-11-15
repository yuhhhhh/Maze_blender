# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 5:48 下午
# @Author  : yuhang
# @FileName: maze_env.py
# @Software: PyCharm
# @Email   : yuhang.1109@qq.com

import bpy
import time
import numpy as np

s = np.array([0, 0])
start = np.array([2,0])
end = np.array([6,7])

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

class Maze():
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.buildMaze()


    def buildMaze(self):

        for i in range(8):
            for j in range(8):
                if maze[i, j] == 1:
                    bpy.ops.mesh.primitive_cube_add(location=(2 * i, 2 * j, 0))
        bpy.ops.mesh.primitive_uv_sphere_add(location=(2 * start[0], 2 * start[1], 2))
        s = start

    def reset(self):
        time.sleep(0.5)
        bpy.data.objects["Sphere"].location = (2 * start[0], 2 * start[1], 2)
        s = start
        return s


    def step(self, action):
        base_action = np.array([0, 0])
        if action == 0: # up
            if s[0] != 0:
                base_action[0] -= 1
        if action == 1: # down
            if s[0] != 7:
                base_action[0] += 1
        if action == 2: # rigth
            if s[1] != 7:
                base_action[1] += 1
        if action == 3: # left
            if s[1] != 0:
                base_action[1] -= 1

        bpy.data.objects["Sphere"].location = (2 * base_action[0], 2 * base_action[1], 2)
        s_ = base_action

        if s_ == end:
            reward = 1
            done = True
            s_ = 'terminal'

        if maze[s_[0], s_[1]] == 0:
            reward = -1
            done = True
            s_ = 'terminal'

        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)