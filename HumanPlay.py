from GameRule import Board
from MCTS_hex import Node, MCTS, UCB1_select

class HumanPlayer:
    def __init__(self, color):
        self.color = color

    def get_move(self, board):
        while True:
            if self.color == 1 and board.get_valid_moves().shape[0] == 121:
                print("Red's first move must be at C1 (1 2).")
                return 1, 2
            move = input("Enter your move in format 'x y': ")
            x, y = map(int, move.split())
            if 0 <= x < 11 and 0 <= y < 11 and board.board[x, y] == 0:
                return x, y
            else:
                print("Invalid move. Please enter again.")

class MCTSPlayer:
    def __init__(self, iterations):
        self.iterations = iterations

    def get_move(self, board):
        root = Node(board)
        best_child = MCTS(root, self.iterations)
        return best_child.action
    
def print_board(board):
    print("  ", end="")
    for i in range(board.size):
        print(chr(ord('A') + i), end=" ")
    print()
    for i in range(board.size):
        print(" " * i, end="")
        print(i, end=" ")
        print(" ".join("\033[31mR\033[0m" if x == 1 else "\033[34mB\033[0m" if x == -1 else '.' for x in board.board[i]))
    print()

def MCTS_main():
    board = Board()
    players = [MCTSPlayer(1000), MCTSPlayer(1000)]
    print_board(board)
    while True:
        for player in players:
            x, y = player.get_move(board)
            board.move(x, y)
            print_board(board)
            game_over, winner = board.is_game_over()
            if game_over:
                if winner == 1:
                    print("Red wins!")
                elif winner == -1:
                    print("Blue wins!")
                else:
                    print("Draw!")
                return

def HumanMain():
    board = Board()
    players = [HumanPlayer(1), HumanPlayer(-1)]
    print_board(board)
    while True:
        for player in players:
            x, y = player.get_move(board)
            board.move(x, y)
            print_board(board)
            game_over, winner = board.is_game_over()
            if game_over:
                if winner == 1:
                    print("Red wins!")
                elif winner == -1:
                    print("Blue wins!")
                else:
                    print("Draw!")
                return

if __name__ == "__main__":
    MCTS_main()
