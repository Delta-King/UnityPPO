# -*- coding: gb18030 -*-
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
from ppo import Agent
from tensorboardX import SummaryWriter


def main():
    writer = SummaryWriter(comment="-" + "3DBall_PPO")
    env_directory = 'UnityEnvironment'
    unity_env = UnityEnvironment(env_directory, base_port=5005, no_graphics=True)
    env = UnityToGymWrapper(unity_env, uint8_visual=True)
    N = 12000
    batch_size = 64
    n_epochs = 3
    alpha = 0.0003
    agent = Agent(action_dim=2, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=8, gae_lambda=0.95)
    max_step = 5000000

    score_history = []

    learn_iters = 0
    n_episode = 0
    n_steps = 0

    # 初始化环境
    observation = env.reset()
    score = 0

    for i in range(max_step):
        action, log_prob = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, log_prob, observation_, reward, done)
        if n_steps % N == 0:
            alpha_ = alpha * (1 - n_steps / max_step) + 1e-10
            actor_loss, critic_loss, entropy, total_loss = agent.learn(alpha_)
            learn_iters += 1
            # alpha -= 0.00001
            # alpha = max(alpha, 0.0003)
            # writer.add_scalar("avg score", avg_score, learn_iters)
            writer.add_scalar("actor loss", actor_loss, learn_iters, display_name='actor loss')
            writer.add_scalar("critic loss", critic_loss, learn_iters, display_name='critic loss')
            writer.add_scalar("total loss", total_loss, learn_iters, display_name='total loss')
            writer.add_scalar("entropy", entropy, learn_iters, display_name='entropy')

        observation = observation_

        if done:
            n_episode += 1
            score_history.append(score)
            observation = env.reset()
            avg_score = np.mean(score_history[-100:])
            std_score = np.std(score_history[-100:])
            print('episode', n_episode, 'score %.2f' % score, 'avg score %.2f' % avg_score, 'std score %.2f' % std_score,
                  'time_steps', n_steps, 'learning_steps', learn_iters)
            writer.add_scalar("score", score, i)
            writer.add_scalar("avg score", avg_score, i)
            writer.add_scalar("std score", std_score, i)
            score = 0

        # if avg_score > best_score:
        #     best_score = avg_score

        # writer.add_scalar("actor loss", actor_loss, learn_iters)
        # writer.add_scalar("critic loss", critic_loss, learn_iters)


if __name__ == '__main__':
    main()
