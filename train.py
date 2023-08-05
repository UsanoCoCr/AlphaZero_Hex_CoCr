from __future__ import print_function
import numpy as np
import random
from collections import defaultdict, deque
from GameRule import Board
from HumanPlay import VirtualGame
from MCTS_hex import MCTSPlayer as MCTS_Pure
from MCTS_alphahex import MCTSPlayer
from ActorCriticNetwork import PolicyValueNet

class Train():
    def __init__(self, init_model=None):
        self.board_width = 11
        self.board_height = 11
        self.board = Board()
        self.game = VirtualGame()

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 自适应调整学习率
        self.temp = 1.0  # 温度参数
        self.c_puct = 0.5
        self.buffer_size = 10000
        self.batch_size = 512  # 批量训练的大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # 每次更新的训练步数
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        self.n_playout = 400  # 搜索次数
        self.pure_mcts_playout = 1000 # 纯蒙特卡洛树搜索次数

        if init_model:
            # 接续训练
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model, use_gpu=True)
        else:
            # 重新训练
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, use_gpu=True)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, iterations=self.n_playout, is_selfplay=1)

    def get_equi_data(self, play_data):
        # 数据增强
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data
    
    def collect_selfplay_data(self, n_games=1):
        # 收集自我对弈数据
        for i in range(n_games):
            winner, play_data = self.game.self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        # 更新策略网络
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs*np.log(old_probs/new_probs), axis=1))
            if kl > self.kl_targ*4:
                break
        # 自适应调整学习率
        if kl > self.kl_targ*2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ/2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(np.array(winner_batch)-old_v.flatten())/np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch)-new_v.flatten())/np.var(np.array(winner_batch))
        print("kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy
    
    def policy_evaluate(self, n_games=10):
        # 与纯蒙特卡洛树搜索对弈，评估策略网络
        win_cnt = defaultdict(int)
        for i in range(n_games):
            color = i % 2 * 2 - 1
            current_mcts_player = MCTSPlayer(color, self.policy_value_net.policy_value_fn, c_puct=self.c_puct, iterations=self.pure_mcts_playout)
            pure_mcts_player = MCTS_Pure(-color, c_puct=0.5, iterations=self.pure_mcts_playout)
            if color == 1:
                winner = self.game.play(pure_mcts_player, current_mcts_player)
            else:
                winner = self.game.play(current_mcts_player, pure_mcts_player)
            win_cnt[winner] += 1

        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1])/n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_playout, win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio
    
    def run(self):
        # 训练主函数
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # 每隔一定步数评估一次策略网络
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # 更新最优策略
                        self.policy_value_net.save_model('./best_policy.model')
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout < 5000:
                            self.pure_mcts_playout += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    training_pipeline = Train()
    training_pipeline.run()