import model
import wrappers

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn

DEFAULT_ENV_NAME = "BreakoutNoFrameskip-v4"

class Agent:
    def __init__(self, env):
        self.env = env
        self._reset()

    def _reset(self):
        self.state = self.env.reset()

    def play_step(self, net, epsilon=0.0, device="cpu"):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        self.state, reward, is_done, _ = self.env.step(action)

        return reward, is_done

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Use CUDA device")
    parser.add_argument("--fps", type=int, default=30, help="Maximum FPS")

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    agent = Agent(wrappers.make_env(DEFAULT_ENV_NAME))

    net = model.DQN(agent.env.observation_space.shape, agent.env.action_space.n).to(device)
    # state_dict = torch.load("./BreakoutNoFrameskip-v4-best.dat")
    # net.load_state_dict(state_dict)

    frame_idx = 0
    ts_frame = 0.0
    ts = time.time()
    best_reward = None

    while True:
        frame_idx += 1

        reward, is_done = agent.play_step(net, device=device, epsilon=0.02)

        agent.env.render()

        time.sleep(0.01)

        if is_done:
            print("Reward: %d" % (reward))
            break

    agent.env.close()

if __name__ == "__main__":
    main()
