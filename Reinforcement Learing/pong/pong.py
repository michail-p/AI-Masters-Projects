import argparse

import gym
from ma_gym.wrappers import Monitor
# import matplotlib.pyplot as plt # plotting dependency

from PIL import Image

from Agent import Agent
from RandomAgent import RandomAgent

"""
Based on:
https://github.com/koulanurag/ma-gym/blob/master/examples/random_agent.py

This script executes the Pong simulator with two agents, both of which make
use of the ``Agent`` class. Nothing in this file needs to be changed, but
you can make changes for debugging purposes.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pong simulator for ma-gym')
    parser.add_argument('--env', default='PongDuel-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=550,
                        help='episodes (default: %(default)s)')
    parser.add_argument('--render', action='store_true',
                        help='Render to a window (slow; can be unstable on some setups)')
    parser.add_argument('--record', action='store_true',
                        help='Record videos to recordings/ (requires ffmpeg)')
    parser.add_argument('--gif', action='store_true',
                        help='Save a GIF of the last episode (no ffmpeg required)')
    parser.add_argument('--gif-path', default='pong.gif',
                        help='Path for the output gif (default: %(default)s)')
    args = parser.parse_args()

    # Set up environment
    env = gym.make(args.env)
    if args.record:
        env = Monitor(env, directory='recordings/' + args.env, force=True)
    action_meanings = env.get_action_meanings()
    print(env.observation_space[0].shape)
    # Initialize agents
    my_agent = Agent(0, env.action_space[0].n, env.observation_space[0].shape)
    agents = [
        my_agent,
        RandomAgent(1)
    ]

    print(f'Action space: {env.action_space}')
    print(f'Observation (state) space: {env.observation_space}')

    wins = []
    losses = []
    win_loss_history = []
    # Run for a number of episodes
    for ep_i in range(args.episodes):
        are_done = [False for _ in range(env.n_agents)]
        ep_rewards = [0, 0]

        gif_frames = []
        capture_gif = args.gif and (ep_i == args.episodes - 1)

        env.seed(ep_i)
        prev_observations = env.reset()
        if args.render:
            env.render()

        while not all(are_done):
            # Observe:
            prev_observations = env.get_agent_obs()
            actions = []
            # For each agent, act:
            for (index, observation) in enumerate(prev_observations):
                action = agents[index].act(prev_observations)
                actions.append(action)
                # Use the command below to print the exact actions that the
                # agents are executing:
                # print([action_meanings[index][action] for action in actions])
            # Trigger the actual execution
            observations, rewards, are_done, infos = env.step(actions)
            # For each agent, update observations and rewards
            for agent in agents:
                agent.update_history(
                    prev_observations,
                    actions[agent.id],
                    rewards[agent.id],
                    observations
                )
            # Debug: print obs, rewards, info
            # print(observations, rewards, infos)
            # Rewards are either 0 or [1, -1], or [-1, 1], try out by out-commenting the line below
            # Note that your agent is agent 0
            # if not (rewards[0] == 0 and rewards[1] == 0): print(rewards)
            for (index, reward) in enumerate(rewards):
                ep_rewards[index] += reward
            if args.render:
                env.render()
            if capture_gif:
                frame = env.render(mode='rgb_array')
                gif_frames.append(frame)
        # Aggregate wins and losses
        bottom_line = ep_rewards[0]
        if bottom_line < 0:
            wins.append(0)
            losses.append(abs(bottom_line))
        else:
            wins.append(bottom_line)
            losses.append(0)
        win_loss_history.append(sum(wins) - sum(losses))
        print('Episode #{} Rewards: {}'.format(ep_i, ep_rewards))
        print(f'Wins - losses: {sum(wins) - sum(losses)}')
        print(f'Epsilon: {my_agent.epsilon}')
        print(f'Q table size: {len(my_agent.q)}')
        if len(wins) > 10:
            print(f'Last 10 games: {sum(wins[-10:]) - sum(losses[-10:])}')

        if capture_gif and gif_frames:
            images = [Image.fromarray(frame) for frame in gif_frames]
            images[0].save(
                args.gif_path,
                save_all=True,
                append_images=images[1:],
                duration=33,
                loop=0,
            )
            print(f'Saved gif to: {args.gif_path}')
        # remove comments for a primitive plot; also remove comment for dependency (line 5)!
        # plt.clf()
        # plt.cla()
        # plt.close()
        # plt.plot(win_loss_history)
        # plt.pause(0.1)
    env.close()
