# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 6:41 下午
# @Author  : yuhang
# @FileName: run_agent.py
# @Software: PyCharm
# @Email   : yuhang.1109@qq.com
"""
from maze_env import Maze
from RL_brain import QLearningTable
"""
import bpy
import time
import numpy as np
import pandas as pd

location = np.array([2, 0])
end = np.array([6, 7])

maze = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 0, 0, 1, 0, 1, 0],
                [0, 1, 1, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1, 1, 1, 0],
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
                if maze[i, j] == 0:
                    bpy.ops.mesh.primitive_cube_add(location=(2 * i, 2 * j, 0))
        bpy.ops.mesh.primitive_uv_sphere_add(location=(2 * 2, 2 * 0, 0))
        bpy.ops.mesh.primitive_plane_add(size = 16, location = (7, 7, -1))

    def reset(self):
        global location
        time.sleep(0.5)
        bpy.data.objects["Sphere"].location = (2 * 2, 2 * 0, 0)
        location = np.array([2, 0])
        return location


    def step(self, action):
        global location
        lasts = np.array([0, 0])
        lasts[0] = location[0]
        lasts[1] = location[1]
        if action == 0: # up
            if location[0] != 0:
                location[0] -= 1
        elif action == 1: # down
            if location[0] != 7:
                location[0] += 1
        elif action == 2: # rigth
            if location[1] != 7:
                location[1] += 1
        elif action == 3: # left
            if location[1] != 0:
                location[1] -= 1



        if location[0] == end[0] and location[1] == end[1]:
            print('Congratulation!')
            reward = 50
            done = True
            bpy.data.objects["Sphere"].location = (2 * location[0], 2 * location[1], 0)
            location = 'terminal'

        elif maze[location[0], location[1]] == 0:
            # print('2')
            reward = -10
            done = False
            location[0] = lasts[0]
            location[1] = lasts[1]

        else:
            # print('3')
            reward = -0.1
            done = False
            bpy.data.objects["Sphere"].location = (2 * location[0], 2 * location[1], 0)

        return location, reward, done

    def render(self):
        time.sleep(0.1)

class QLearningTable:
    def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

def update():
    for episode in range(100):
        observation = np.array([2, 0])
        env.reset()

        while True:
            env.render()

            action = RL.choose_action(str(observation))

            observation_, reward, done = env.step(action)
            # 动作执行后，更新q值表
            RL.learn(str(observation), action, reward, str(observation_))
            if episode == 99:
                print(observation)
                print(action)
                print(observation_)
            if done:
                break
            else:
                observation[0] = observation_[0]
                observation[1] = observation_[1]

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))
    update()
    print(RL.q_table)