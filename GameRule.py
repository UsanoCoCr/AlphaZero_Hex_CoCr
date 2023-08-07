import numpy as np

class Board:
    def __init__(self):
        self.size = 11
        self.board = np.zeros((self.size, self.size))
        self.player = 1  # 1 for red, -1 for blue

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.player = 1
        self.last_move = -1

    def move(self, x, y):
        assert self.board[x, y] == 0, "Invalid move"
        self.board[x, y] = self.player
        self.player *= -1
        self.last_move = x * self.size + y

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def is_game_over(self):
        # Check for red win
        for i in range(self.size):
            if self._has_path((0, i), 1):
                return True, 1

        # Check for blue win
        for i in range(self.size):
            if self._has_path((i, 0), -1):
                return True, -1

        # Check for draw
        if len(self.get_valid_moves()) == 0:
            return True, 0

        return False, None

    def _has_path(self, start, player):
        stack = [start]
        visited = set()
        while stack:
            x, y = stack.pop()
            if player == 1 and x == self.size:
                return True
            if player == -1 and y == self.size:
                return True
            if x == self.size or y == self.size or self.board[x, y] != player:
                continue
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx <= self.size and 0 <= ny <= self.size and (nx, ny) not in visited:
                    stack.append((nx, ny))
                    visited.add((nx, ny))
        return False
    
    def print_board(self):
        print("  ", end="")
        for i in range(self.size):
            print(chr(ord('A') + i), end=" ")
        print()
        for i in range(self.size):
            print(" " * i, end="")
            print(i, end=" ")
            print(" ".join("\033[31mR\033[0m" if x == 1 else "\033[34mB\033[0m" if x == -1 else '.' for x in self.board[i]))
        print()

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.size, self.size))
        if not np.all(self.board == 0):
            #问题出在建模上，我的board是二维np数组，原作者的state是字典，需要修改为字典
            player1_moves = np.array(np.where(self.board == 1)).T
            player2_moves = np.array(np.where(self.board == -1)).T
            moves = np.concatenate((player1_moves, player2_moves))
            players = np.array([1]*len(player1_moves) + [-1]*len(player2_moves))
            move_curr = moves[players == self.player]
            move_oppo = moves[players != self.player]
            square_state[0][move_curr // self.size,
                            move_curr % self.size] = 1.0
            square_state[1][move_oppo // self.size,
                            move_oppo % self.size] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.size,
                            self.last_move % self.size] = 1.0
        if self.board.sum() % 2 == 0:
            square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]