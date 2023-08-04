from GameRule import Board
from MCTS_hex import MCTSPlayer
import numpy as np
import copy

class HumanPlayer:
    def __init__(self, color):
        self.color = color

    def get_action(self, board):
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

    def set_player_ind(self, p):
        self.player = p

def play(player1, player2, startplayer=1):
    board = Board()
    board.reset()
    players = {1: player1, -1: player2}
    player1.set_player_ind(1)
    player2.set_player_ind(-1)
    board.player = startplayer
    while True:
        current_player = board.player
        player_in_turn = players[current_player]
        action = player_in_turn.get_action(board)
        board.move(action[0], action[1])
        board.print_board()
        game_over, winner = board.is_game_over()
        if game_over:
            if winner == 1:
                print("Red wins!")
            elif winner == -1:
                print("Blue wins!")
            else:
                print("Draw!")
            break

if __name__ == "__main__":
    print("game start")
    player1 = HumanPlayer(1)
    player2 = MCTSPlayer(-1, c_puct=0.5, iterations=1000)
    play(player1, player2)