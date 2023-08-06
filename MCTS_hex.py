import numpy as np
import copy
from GameRule import Board
from operator import itemgetter

def policy_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    valid_moves = board.get_valid_moves()
    action_probs = np.ones(len(valid_moves))/len(valid_moves)
    return zip(map(tuple, valid_moves), action_probs), 0

class Node:
    def __init__(self, parent=None, prior=1):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.Q = 0
        self.U = 0
        self.prior = prior
    
    def is_leaf(self):
        return self.children == {}
    
    def is_root(self):
        return self.parent == None

    def update(self, result):
        self.visits += 1
        self.Q += (result - self.Q) / self.visits

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
            It is a combination of leaf evaluations Q, and this node's prior
            adjusted for its visit count, u.
            c_puct: a number in (0, inf) controlling the relative impact of
                value Q, and prior probability P, on this node's score.
        """
        self.U = c_puct * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)
        return self.Q + self.U
    
    def select(self, c_puct):
        """select action among children that gives maximum action value Q
        plus bonus u(P).
        return: a node from where search will continue.
        """
        action, node = max(self.children.items(), key=lambda item: item[1].get_value(c_puct))
        return action, node

    
    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(parent=self, prior=prob)

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
        
        action_probs, _ = self.policy(board)
        game_over, winner = board.is_game_over()
        if not game_over:
            node.expand(action_probs)
        leaf_value = self.simulate(board)
        node.backpropagation(-leaf_value)

    def simulate(self, board, limit=200):
        player = board.player
        for i in range(limit):
            game_over, winner = board.is_game_over()
            if game_over:
                break
            valid_moves = board.get_valid_moves()
            action_probs = [(action, 1.0 / len(valid_moves)) for action in valid_moves]
            p_action = max(action_probs, key=itemgetter(1))[0]
            board.move(p_action[0], p_action[1])
        else:
            print("WARNING: rollout reached move limit")
        if winner == 0:
            return 0
        else:
            return 1 if winner == player else -1
        
    def get_move(self, board):
        for _ in range(self.iterations):
            v_board = copy.deepcopy(board)
            self.mcts_search(v_board)
        """ best_child = max(self.root.children.values(), key=lambda node: )
        for action, node in self.root.children.items():
            if node is best_child:
                return action """
        action, node = max(self.root.children.items(), key=lambda item: item[1].get_value(self.c_puct))
        return action
        
    def update_move(self, last_action):
        if last_action in self.root.children:
            self.root = self.root.children[last_action]
            self.root.parent = None
        else:
            self.root = Node()

class MCTSPlayer:
    def __init__(self, c_puct=0.5, iterations=1000):
        self.mcts = MCTS(policy_fn, c_puct, iterations)

    def set_player_ind(self, p):
        self.player = p
    
    def reset_player(self):
        self.mcts.update_move(-1)

    def get_action(self, board):
        if board.get_valid_moves().shape[0] == 121:
            return 1, 2
        sensible_moves = board.get_valid_moves()
        if len(sensible_moves) > 0:
            action = self.mcts.get_move(board)
            self.mcts.update_move(-1)
            return action
        else:
            print("WARNING: the board is full")
