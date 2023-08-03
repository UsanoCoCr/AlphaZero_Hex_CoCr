# AlphaHex结构简介

## 1. 项目结构
    
    ```
    AlphaHex
    ├── GameRule.py : 游戏状态建模
    ├── HumanPlay.py : 支持人机、机机对战的TUI界面
    ├── MCTS_hex.py : 单纯的MCTS算法实现
    ├── train.py : 神经网络的训练实现
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