import numpy as np
import copy
import torch
from GameRule import Board
from operator import itemgetter

softmax = torch.nn.Softmax(dim=1)

class Node:
    def __init__(self, parent=None, prior=1):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.Q = 0.0
        self.U = 0.0
        self.prior = prior
    
    def is_leaf(self):
        return self.children == {}
    
    def is_root(self):
        return self.parent == None

    def update(self, result):
        self.visits += 1
        self.Q += (result - self.Q) / self.visits

    def get_value(self, c_puct):
        self.U = c_puct * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)
        return self.Q + self.U
    
    def select(self, c_puct):
        action, node = max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
        return action, node
    
    def expand(self, action_priors):
        for action, prob in action_priors:
            action_tuple = tuple(action.tolist())
            if action_tuple not in self.children:
                self.children[action_tuple] = Node(parent=self, prior=prob)

    def backpropagation(self, result):
        """backpropagation update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        if self.parent:
            self.parent.backpropagation(-result)
        self.update(result)

class MCTS:
    def __init__(self, policy, c_puct=0.5, iterations=1000):
        self.root = Node()
        self.policy = policy
        self.c_puct = c_puct
        self.iterations = iterations

    def mcts_search(self, board):
        """simulation runs a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            board.move(action[0], action[1])
        
        action_probs, leaf_value = self.policy(board)
        game_over, winner = board.is_game_over()
        if not game_over:
            node.expand(action_probs)
        else:
            if winner == 0:
                leaf_value = 0
            else:
                leaf_value = 1 if winner == board.player else -1
        node.backpropagation(-leaf_value)

    def get_move(self, board, temp=1e-3):
        for i in range(self.iterations):
            v_board = copy.deepcopy(board)
            self.mcts_search(v_board)

        # 根据访问次数分配概率
        action_visits = [(action, node.visits) for action, node in self.root.children.items()]
        actions, visits = zip(*action_visits)
        tensor = torch.tensor(1.0/temp * np.log(np.array(visits) + 1e-10))
        action_probs = torch.nn.functional.softmax(tensor, dim=0).numpy()
        return actions, action_probs
        
    def update_move(self, last_action):
        # 确保 last_action 是元组类型
        if not isinstance(last_action, tuple):
            last_action = tuple(last_action)
            
        if last_action in self.root.children:
            self.root = self.root.children[last_action]
            self.root.parent = None
        else:
            self.root = Node()


class MCTSPlayer:
    def __init__(self, policy_value_function=None, c_puct=0.5, iterations=1000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, iterations)
        self.is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p
    
    def reset_player(self):
        self.mcts.update_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        if board.get_valid_moves().shape[0] == 121:
            if return_prob == 1:
                move = np.array([1, 2])
                move_probs = np.zeros(board.size**2)
                return move, move_probs
        sensible_moves = board.get_valid_moves()
        move_probs = np.zeros(board.size**2)
        if len(sensible_moves) > 0:
            actions, probs = self.mcts.get_move(board, temp)
            for action, prob in zip(actions, probs):
                index = action[0] * 11 + action[1]
                move_probs[index] = prob
            if self.is_selfplay:
                # 添加噪声提高探索性
                actions_1d = [action[0] * 11 + action[1] for action in actions]
                move_1d = np.random.choice(actions_1d, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                move = np.array([move_1d // 11, move_1d % 11])
                self.mcts.update_move(move)
            else:
                # 选择最大概率的动作
                move = np.random.choice(actions, p=probs)
                self.mcts.update_move(-1)

            if return_prob == 1:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
