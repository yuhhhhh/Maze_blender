# 一些链接

Github地址：[https://github.com/yuhhhhh/Maze_blender](https://github.com/yuhhhhh/Maze_blender)

莫烦强化学习：[https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/](https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/)

Blender官方api：[https://docs.blender.org/api/2.90/info_quickstart.html](https://docs.blender.org/api/2.90/info_quickstart.html)

# 问题介绍

走迷宫是一个很简单的游戏，在一些简单的场景我们可以很轻松的找到终点，对于复杂的场景，花的时间可能会长一点，最近开始学习强化学习，走迷宫是强化学习入门的一个简单案例，本次尝试利用强化学习来训练一个机器人找到走迷宫的最短路径，强化学习部分的代码主要来自于莫烦的走迷宫案例，当然关于q-learning也是在莫烦的网站上学习的。初衷是为了验证blender是否可以进行仿真，也就是我们强化学习的过程是否可以在blender上实时观察到，在网上可以搜到的利用blender仿真的案例相当少，虽然官网给了丰富的api，这也许和blender在国内本来就不火有关，查阅了相关接口确实可以在blender中建立一个迷宫模型，接下来就开始吧，Let's go！

# Q-Learning

走迷宫用到的是Q-Learning算法，关于Q-Learning的讲解网上有相当多的资料可以查询，本文主要说一下我自己的见解，大家参考就行，机器人在每一个状态可以选择4个动作，即up,down,left,right四个动作，假如我们在状态s1，采取动作a1，这样机器人的状态就会发生变化，变为s2。而根据采取的动作a1不同，下一个状态s2就是不一样的，在迷宫中，状态主要对应着机器人的坐标，而在每一个坐标主要有3种状态，即：撞到墙壁，走到终点，正常行走三种状态，每一种状态我们设定一个reward，即奖励值，因为Q-Learning主要通过不断更新来得到最优的q值表，也就是我们在状态s1时选取动作不是完全随机的，会根据q值表选择q值最大的动作，即利益最大化。每个动作执行后我们更新q值表，这样不断反复的更新，q值表就能实现最优，设定的奖励值不是固定的，可以根据具体效果进行更改，比如我选择的奖励值如下：

|状态|奖励值|
|-|-|
|撞到墙壁|-10|
|走到终点|50|
|正常行走|-0.1|


接下来说说设置奖励值的思路，走到终点肯定是我们首要考虑的，所以它应该是一个正的奖励值，且这个值应该很大，因为由于q-learning的特性，我们到终点的这一段路对应状态的q值都会相应增大，撞到墙壁肯定是我们不希望的所以设定为负的，正常行走为什么也设置为负的，因为我们的目的是最短的路径，走的步数多了自然是我们不想看到的。

# Blender库API

本次主要使用的blender的创建物体、移动物体、插帧的api。

首先我们导入blender的库

```Python
import bpy
```

## 创建迷宫

在blender中创建一个迷宫，可以先想好大体的迷宫路线，根据自己想好的迷宫图，转换为对应的矩阵，本文使用的迷宫矩阵如下：

```Python
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
```

可以走的路为1，不可以走的为0，根据0、1就可以创建障碍物了，同样我们机器人的位置坐标发生变化时，把坐标作为矩阵的行列索引，就可以快速知道机器人是否撞到墙壁，是否走到终点。

### 创建物体

#### 障碍物

```Python
bpy.ops.mesh.primitive_cube_add(location=(2 * i, 2 * j, 0)) 
```

上述代码创建了一个cube，作为迷宫的阻碍物，默认设定的cube是2x2x2大小的，

location参数是cube中心的坐标

#### 机器人

```Python
bpy.ops.mesh.primitive_uv_sphere_add(location=(2 * 2, 2 * 0, 0))
```

创建一个球体，这个球体就是走迷宫的机器人，参数和创建cube是一样的

#### 平面

```Python
bpy.ops.mesh.primitive_plane_add(size = 16, location = (7, 7, -1))
```

创建了一个平面，size参数表示平面的大小，因为我们迷宫矩阵设置的是8x8矩阵，每一个方块是2x2x2的，所以一个平面的长为16。

location参数也就是平面中心对应的坐标

创建的迷宫如下：

![](https://secure-static.wolai.com/static/kRHcWQ4J8BSdezLLe17cTs/image.png)

## 移动物体

本次只考虑机器人在二维平面的移动，即x和y坐标的变化，z不变，blender提供了控制物体坐标的api

```Python
 bpy.data.objects["Sphere"].location = (2 * location[0], 2 * location[1], 0)
```

objects后方括号里的是需要移动的物体元素的名称，取得一个对象，location参数也就是对象的坐标，根据机器人当前坐标进行变化

## 插帧

由于blender中物体的显示不能随着代码实时显示，它会在代码跑完了显示出来，每次跑完球体都会在终点位置，无法观察到运动的过程，后来查到官方提供了插帧的api

```Python
obj = bpy.data.objects["Sphere"]
obj.keyframe_insert(data_path="location", frame=tFrame, index=0)
```

首先选择需要记录的物体，即我们的机器人，obj此时就是我们的机器人对象，利用keyframe_insert方法就可以在对应帧记录对象的信息，data_path表示记录的信息，frame表示记录的对应帧，index表示location的元素，0是x轴，1是y轴，2是z轴

# 常见的问题

## blender内的python第三方库不全

由于blender中使用的python是它自带的，可能很多需要用到的库都没有，那么就需要安装库，网上搜blender安装python库有很多解决方法，恰好这些方法在我的电脑上都不适用，但是后来我找到了一种方法

首先我们找到blender自带python的路径，比如我的是：/Applications/Blender.app/Contents/Resources/2.83

然后打开pycharm，pycharm中可以设置解释器，调到对应的blender的解释器，就会显示它的一些库，找到➕号，就可以安装需要的库

## blender报错无法看见

通过命令行启动，启动方式

### Windows

找到blender.exe文件的文件夹，在路径栏输入大写的CMD，就会跳出来一个命令行界面，输入blender即可启动

### Mac

通过终端进入blender目录，找到启动文件，比如我的路径是/Applications/Blender.app/Contents/MacOS

进入这个路径，输入./blender即可启动

# 代码

```Python
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
tFrame = 0

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
    def __init__(self, actions, learning_rate = 0.5, reward_decay = 0.9, e_greedy = 0.9):
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
    global tFrame
    obj = bpy.data.objects["Sphere"]
    obj.keyframe_insert(data_path="location", frame=tFrame, index=0)
    obj.keyframe_insert(data_path="location", frame=tFrame, index=1)
    tFrame += 1
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
                obj.keyframe_insert(data_path = "location", frame = tFrame, index = 0)
                obj.keyframe_insert(data_path = "location", frame = tFrame, index = 1)
                tFrame += 1
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
```