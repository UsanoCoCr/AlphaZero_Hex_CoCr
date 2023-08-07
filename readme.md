# AlphaHex结构简介

## 1. 项目结构
    
    ```
    AlphaHex
    ├── GameRule.py : 游戏状态建模
    ├── HumanPlay.py : 对战接口
    ├── MCTS_hex.py : 单纯的MCTS算法实现
    ├── train.py : 神经网络的训练实现
    ├── MCTS_alphahex.py : 插入神经网络的MCTS算法实现
    ├── ActorCriticNetwork.py : ActorCritic网络的搭建
    └── README.md
    ```

## 2. 游戏规则
- 棋盘大小为11*11
- 红蓝双方交替落子，红方先手
- 红方第一手必须落在棋盘的C1位置（Hex棋先手优势过大，所以强制规定第一手C1以限制先手优势。C1是强软研究得到的最平衡点）
- 红连接上-下棋盘判红胜利，蓝连接左-右棋盘判蓝胜利，没有平局

## 3. 更新日志
### 2023.8.3
- 完成GameRule.py，实现了游戏状态的建模
- 实现了HumanPlay.py的基础设置，支持人类玩家游玩
- 实现了MCTS_hex.py,完成了Python版本的基本MCTS算法
- 初步探索train.py的实现

### 2023.8.4
- 对MCTS_hex.py完成了重构，优化代码结构
- 对MCTS_hex.py进行了优化，添加接入神经网络的接口
- 修改了GameRule.py的错误

### 2023.8.5
- 增加了MCTS_alphahex.py，插入神经网络的MCTS算法，但出现policy=None的错误
- 增加了train.py
- 增加了ActorCriticNetwork.py，实现了ActorCritic网络的搭建
- 改进了GameRule.py，增加VirtualGame类，实现了抽象对局

### 2023.8.6
- 修改了policy=None的错误
- 发现board存储格式影响训练过程中算法，待添加dictionary存储格式

### 2023.8.7
- 修改游戏建模，放弃添加存储格式，但增加了运行时间
- 完成train.py，但模拟对局速度慢，可能策略：将mcts放至gpu