import model
import wrappers

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10**4             # size of the replay buffer
LR = 1e-4                       # learning rate used by the optimizer
SYNC_TARGET_FRAMES = 10**3      # sync target network every k frames 
REPLAY_START_SIZE = 10**4       # frames to wait before populating the replay buffer


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done',
                                                               'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.permutation(len(self.buffer))[:batch_size]
        states, actions, rewards, dones, new_state = zip(*[self.buffer[i] for i in indices])
        return  np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
                np.array(dones, dtype=np.uint8), np.array(new_state)

class ExperienceSource:
    def __init__(self, env, agent, steps_count=2):
        assert isinstance(env, gym.Env)
        assert isinstance(agent, model.DQN)
        assert isinstance(steps_count, int)

        self.env = env
        self.agent = agent
        self.steps_count = steps_count

    def __iter__(self):
        state, action, reward, new_state = [], [], [], []


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # select action (random or from net depending on greediness)
        # if np.random.random() < epsilon:
            # action = self.env.action_space.sample()
        # else:
        state_a = np.array([self.state], copy=False)    # [] adds dimension for batch (?)
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())

        # make step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # save experience in the experience buffer
        experience = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(experience)
        self.state = new_state

        # reset environment and return total reward if finished
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    # create Tensors and move to GPU if available
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    # Q values predictions with net
    state_action_value = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # Values of the most valued states predicted by the target net
    argmax_v = net(next_states_v).max(dim=1)[1]
    tgt_q = tgt_net(next_states_v).gather(1, argmax_v.unsqueeze(-1)).squeeze(-1)
    tgt_q[done_mask] = 0.0              # if this is not done, training will not converge
    tgt_q =tgt_q.detach()  # target network is not trained

    # Bellman equation
    expected_state_action_values = rewards_v + GAMMA * tgt_q
    return nn.MSELoss()(state_action_value, expected_state_action_values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Environment to train on. Default =\
                        " + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND, help="Mean reward\
                        boundary to stop training. Default = %.2f" % MEAN_REWARD_BOUND)
    parser.add_argument("--play_steps", type=int, default=4, help="Number of plays each step (increases batch size)")
    args = parser.parse_args()

    # use CUDA if asked
    device = torch.device("cuda" if args.cuda else "cpu")

    # create the environment
    env = wrappers.make_env(args.env)

    # Create net and target net
    net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    # load saved model
    # state_dict = torch.load("./BreakoutNoFrameskip-v4-best.dat")
    # net.load_state_dict(state_dict)
    # tgt_net.load_state_dict(state_dict)

    # Create Tensorboard writer
    writer = SummaryWriter(comment="-" + args.env)

    # print the net architecture for feedback
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)

    optimizer = optim.Adam(net.parameters(), lr=LR)

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    # training loop
    while True:
        frame_idx += args.play_steps

        for _ in range(args.play_steps):
            reward = agent.play_step(net, device=device)
            # agent.env.render()

            # if episode ended
            if reward is not None:
                # save reward of the episode
                total_rewards.append(reward)

                # calculate frames per second
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts = time.time()
                ts_frame = frame_idx

                # calculate mean reward over the 100 last episodes
                mean_reward = np.mean(total_rewards[-100:])

                # log
                print("Frame: %d\tDone games: %d\tMean reward: %.3f\tFPS: %.2f" %
                      (frame_idx, len(total_rewards), mean_reward, speed))
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(net.state_dict(), args.env + "-best.dat")
                    # if there was a previous best model
                    if best_mean_reward is not None:
                        print("Best mean reward updated: %.3f -> %.3f. Model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward

        # Do not train until some frames have passed. This helps the model converge, but it is not
        # clear why.
        if frame_idx < REPLAY_START_SIZE:
            continue

        # sync target network with net every SYNC_TARGET_FRAMES frames
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE * args.play_steps)
        loss = calc_loss(batch, net, tgt_net, device)
        loss.backward()
        optimizer.step()

    writer.close()

if __name__ == "__main__":
    main()
