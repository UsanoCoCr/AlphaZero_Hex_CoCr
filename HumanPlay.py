from GameRule import Board
from MCTS_hex import MCTSPlayer
import numpy as np
import copy

class HumanPlayer:
    def __init__(self, color):
        self.color = color

    def get_action(self, board):
        while True:
            if board.get_valid_moves().shape[0] == 121:
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

class VirtualGame:
    def __init__(self):
        self.board = Board()
        self.board.reset()
        self.board.player = 1
    
    def reset(self):
        self.board.reset()
        self.board.player = 1

    def play(self, player1, player2, startplayer=1):
        self.board.reset()
        self.players = {1: player1, -1: player2}
        player1.set_player_ind(1)
        player2.set_player_ind(-1)
        self.board.player = startplayer
        while True:
            current_player = self.board.player
            player_in_turn = self.players[current_player]
            action = player_in_turn.get_action(self.board)
            self.board.move(action[0], action[1])
            game_over, winner = self.board.is_game_over()
            if game_over:
                if winner == 1:
                    return 1
                elif winner == -1:
                    return -1
                else:
                    return 0
                
    def self_play(self, player, temp=1e-3):
        self.reset()
        self.players = {1: player, -1: player}
        self.board.player = 1
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # 存储数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.player)
            # 执行动作
            #print("move is: ", move)
            #print("move_probs is: ", move_probs)
            self.board.move(move[0], move[1])
            game_over, winner = self.board.is_game_over()
            if game_over:
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置MCTS根节点
                player.reset_player()
                return winner, zip(states, mcts_probs, winners_z)
            

if __name__ == "__main__":
    print("game start")
    player1 = HumanPlayer(1)
    player2 = MCTSPlayer(-1, c_puct=0.5, iterations=1000)
    play(player1, player2)