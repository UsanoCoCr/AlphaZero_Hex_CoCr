import numpy as np
import copy
from GameRule import Board

class Node:
    def __init__(self, board, action=None, parent=None):
        self.board = board
        self.action = action
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = board.get_valid_moves()
    
    def update_untried_actions(self):
        self.untried_actions = self.board.get_valid_moves()
        return self.untried_actions

    def add_child(self, action, board):
        child = Node(board, action, parent=self)
        self.untried_actions = np.delete(self.untried_actions, np.where((self.untried_actions == action).all(axis=1)))
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

def MCTS(root, iterations):
    for _ in range(iterations):
        node = root
        board = copy.deepcopy(root.board)

        # Selection
        while node.update_untried_actions().size == 0 and not board.is_game_over()[0]:
            node = UCB1_select(node)
            if node.action is not None:
                board.move(*node.action)

        if node.untried_actions.shape[0] > 0:
            index = np.random.randint(node.untried_actions.shape[0])
            a = node.untried_actions[index]
            #print("a is: ",a)
            board.move(a[0], a[1])
            node = node.add_child(a, board)

        # Simulation
        while not board.is_game_over()[0]:
            actions = board.get_valid_moves()
            a = actions[np.random.randint(actions.shape[0])]
            #print("a is: ",a)
            board.move(a[0], a[1])
        #print("game over")

        # Backpropagation
        game_over, winner = board.is_game_over()
        while node:
            node.update(winner)
            node = node.parent
    return UCB1_select(root, exploit=True)

def UCB1_select(node, exploit=False):
    C = 0.5
    if exploit:
        C = 0
    best_score = -np.inf
    best_child = None
    for child in node.children:
        score = child.wins / child.visits + C * np.sqrt(2 * np.log(node.visits) / child.visits)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child
