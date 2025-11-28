# env_runner.py

import os
import time
import numpy as np
import torch
from runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class EnvRunner(Runner):

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        # 传入logger
        self.logger = config["logger"]

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # 采样动作
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # 在环境中执行动作并观察奖励和下一个观测
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = (obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic)

                # 将数据插入缓冲区
                self.insert(data)

            # 计算回报并更新网络
            self.compute()
            train_infos = self.train()

            # 后处理
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # 保存模型
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # 记录日志信息
            if episode % self.log_interval == 0:
                end = time.time()
                self.logger.info(
                    f"\n Updates {episode}/{episodes} episodes, total num timesteps {total_num_steps}/{self.num_env_steps}, FPS {int(total_num_steps / (end - start))}.\n"
                )
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                self.logger.info(f"Average episode rewards is {train_infos['average_episode_rewards']:.3f}")
                self.log_train(train_infos, total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )
        
        actions_env = actions

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        
        #
        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)


        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            active_masks=active_masks
        )
    
    # 测试模式下的执行函数
    @torch.no_grad()
    def render(self):
        self.logger.info(f"--- Starting Evaluation Mode for Test Policy: {self.all_args.test_policy} ---")
        
        total_global_rounds = self.all_args.epochs
        self.logger.info(f"Executing for a total of {total_global_rounds} SFL global rounds.")

        # 1. 在整个测试开始前，只 reset 一次环境
        obs = self.envs.reset()
        
        # 2. 【关键】为智能体初始化一个持久的 "记忆" (RNN状态)
        # 这个 rnn_states 变量将在整个测试过程中被持续更新和传递
        rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        # 在测试模式下，critic的RNN状态通常不影响决策，可以保持为零
        rnn_states_critic = np.zeros_like(rnn_states)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # 3. 主测试循环：连续执行所有SFL全局轮次
        for global_round in range(total_global_rounds):
            self.logger.info(f"\n[Runner] === Executing SFL Global Round [{global_round + 1}/{total_global_rounds}] ===")
            
            # A. 准备中心化观测
            if self.use_centralized_V:
                share_obs = obs.reshape(self.n_rollout_threads, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            else:
                share_obs = obs
            
            # B. 使用策略网络, 传入【上一轮更新后】的 rnn_states
            self.trainer.prep_rollout()
            _, actions, _, next_rnn_states, _, _ = self.trainer.policy.get_actions(
                np.concatenate(share_obs),
                np.concatenate(obs),
                np.concatenate(rnn_states), # <--- 使用上一轮的记忆
                np.concatenate(rnn_states_critic),
                np.concatenate(masks),
                deterministic=True # 测试时使用确定性动作
            )

            actions_env = np.array(np.split(_t2n(actions), self.n_rollout_threads))
            
            # C. 【关键】保存智能体为下一轮准备的【新记忆】
            rnn_states = np.array(np.split(_t2n(next_rnn_states), self.n_rollout_threads))
            
            # D. 在环境中执行一个step
            next_obs, rewards, dones, infos = self.envs.step(actions_env)
            
            self.logger.info(f"[Runner] Round [{global_round + 1}] completed. Avg Reward this step: {np.mean(rewards):.4f}")

            # E. 更新观测，为下一轮决策做准备
            obs = next_obs
            
            # 如果环境内部发出了done信号 (例如云端判断收敛)，则提前结束
            if np.all(dones):
                self.logger.info("Environment signaled 'done'. Stopping the test run.")
                break

        self.logger.info("\n--- Evaluation run finished. ---")
        self.envs.close()