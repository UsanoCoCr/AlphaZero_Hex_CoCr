import numpy as np

class Board:
    def __init__(self):
        self.size = 11
        self.board = np.zeros((self.size, self.size))
        self.player = 1  # 1 for red, -1 for blue

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.player = 1

    def move(self, x, y):
        assert self.board[x, y] == 0, "Invalid move"
        self.board[x, y] = self.player
        self.player *= -1

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def is_game_over(self):
        # Check for red win
        for i in range(self.size):
            if self._has_path((i, 0), 1):
                return True, 1

        # Check for blue win
        for i in range(self.size):
            if self._has_path((0, i), -1):
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
            if player == 1 and y == self.size - 1:
                return True
            if player == -1 and x == self.size - 1:
                return True
            if self.board[x, y] != player:
                continue
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in visited:
                    stack.append((nx, ny))
                    visited.add((nx, ny))
        return False
