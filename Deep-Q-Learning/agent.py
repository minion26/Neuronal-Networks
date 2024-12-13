import datetime
import itertools
import os
import random
import flappy_bird_gymnasium
import gymnasium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import yaml


# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

class Agent:

    def __init__(self, hyperparameter_set):
        with open('hyper-params.yaml', 'r') as f :
            all_hyperparams_sets = yaml.safe_load(f)
            hyperparameters = all_hyperparams_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.hidden_dim = hyperparameters['hidden_dim']
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.pth")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.png")


    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
        else:
            last_graph_update_time = datetime.datetime.now()

        # instance of the FlappyBird environment
        # env = gymnasium.make("FlappyBird-v0", render_mode="human"if render else None, use_lidar=False)
        global  best_reward, step_count, memory, target_dqn, epsilon
        # env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)

        # Number of actions
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0] # 12

        rewards_per_episode = []
        epsilon_history = []
        policy_dqn = DQN(num_states, num_actions, self.hidden_dim)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            target_dqn = DQN(num_states, num_actions, self.hidden_dim)
            target_dqn.load_state_dict(policy_dqn.state_dict()) # copy the weights and biases of the policy network to the target network

            step_count = 0

            # policy network optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            best_reward = -999999999
        else :
            epsilon = self.epsilon_min
            memory = ReplayMemory(self.replay_memory_size)
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            policy_dqn.eval()

        for episode in itertools.count():
            # Reset the environment
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float)

            # Initialize the variables
            terminated = False

            episode_reward = 0.0

            while not terminated and episode_reward < 10000:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64)
                else:
                    with torch.no_grad():
                        # just evaluate a state
                        # unsqueeze adds a dimension to the tensor at the specified position, dim = 0 => [1,12]
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                # action.item() to get the value of the tensor
                new_state, reward, terminated, _, info = env.step(action.item())

                # accumulate the reward
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: Episode {episode} - New best reward: {episode_reward}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as f:
                        f.write(log_message + "\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

            # Update graph every x seconds
            current_time = datetime.datetime.now()
            if current_time - last_graph_update_time > datetime.timedelta(seconds=10):
                self.save_graph(rewards_per_episode, epsilon_history)
                last_graph_update_time = current_time


            # decrease epsilon
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min) # makes sure it doesn't go under the minimum
            epsilon_history.append(epsilon)
            #if enough samples in memory (collected enough  experience), sample a mini-batch and train the network
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # copy policy network to target network
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        #optimize using pytorch
        #extract information from the mini-batch into separate tensors
        # mini_batch = memory.sample(self.mini_batch_size) where memory is a tuple of (state, action, new_state, reward, terminated)
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # stack the tensors = concatenate them along a new dimension
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        # tuple of 32 elements , changes from true and false to 1 and 0
        terminations = torch.tensor(terminations).float()

        with torch.no_grad():
            # nu se mai foloseste de if
            # Q_{target} = reward + γ * max_{a'} Q_0(s', a')
            # daca e terminat, 1-1 da 0 => nu se mai aduna nimic
            q_target = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        # policy_dqn(states) has the Q-values for each action in the state (32, 2) where 2 means the number of actions (not flap or flap)
        # if the action is one, will extract the q value with index 1
        q_value = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # compute the loss function using the mean squared error
        # how far the current is far from the target
        # minimize the loss
        loss = self.loss_fn(q_value, q_target)  # (Current Q-value - Q-target)^2 / 2

        # optimize the network => adjust the weights and biases
        self.optimizer.zero_grad()  # zero the gradients
        loss.backward()  # backpropagation
        self.optimizer.step()  # update the weights and biases

        # #look for the experience in the mini-batch that is from the replay memory
        # for state, action, new_state, reward, terminated in mini_batch:
        #     #Q_{target} = reward + γ * max_{a'} Q_0(s', a')
        #     if terminated :
        #         target = reward
        #     else:
        #         with torch.no_grad():
        #             q_target = reward + self.discount_factor_g * target_dqn(new_state).max()
        #
        #     # calculate the current policy prediction of the Q-value
        #     q_value = policy_dqn(state)
        #
            # # compute the loss function using the mean squared error
            # # how far the current is far from the target
            # # minimize the loss
            # loss = self.loss_fn(q_value, q_target) # (Current Q-value - Q-target)^2 / 2
            #
            # #optimize the network => adjust the weights and biases
            # self.optimizer.zer_grad() # zero the gradients
            # loss.backward() # backpropagation
            # self.optimizer.step() # update the weights and biases



if __name__ == '__main__':
    agent = Agent('flappybird1')
    agent.run(is_training=True, render=True)
    print("Done")