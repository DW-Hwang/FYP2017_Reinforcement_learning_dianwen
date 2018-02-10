import gym
import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt


"""
In this exercise, we attempt to use Q-learning to find the optimal policy
to play mountain car. We will do it by creating a grid wall to associate all
discretized continuous state of the game and allocate the Q-value to each states
(with a Q table).
Optimal policy is given by argmax(Q(s,a))
  
## Some information about the game
Game observation is given by [position,velocity]
Position has a range of (-1.2, 0.6)
Velocity has a range of (-0.07, 0.07)
"""
##################################
#####       Q-Learning      ######
##################################

# Create environment
env = gym.make("MountainCar-v0")

# Set our hyper parameters
episodes = 30000
LearnRate = 0.8
gamma = 0.995
reward_history = []
time_step = 200
epsilon = 1 # fixed our e-greedy at 0.05 throughout the game
step_count = 0

# Create grid wall of 100*100 and discretize the continuous state
game_state = []

position_space = pd.cut([-1.2,0.6], bins= 99, retbins= True)[1]
velocity_space = pd.cut([-0.07,0.07], bins= 99, retbins=True )[1]

for position in position_space:
    for velocity in velocity_space:
        game_state.append(np.array([position,velocity]))

game_state = np.array(game_state)

# Initialise Q table
Q = np.zeros((game_state.shape[0], env.action_space.n))


# Create helper that returns the index of the corresponding observation (Q table)
def return_state(observation):
    global game_state
    global position_space
    global velocity_space
    # return index of the nearest position state
    position_index = int(np.digitize(observation[0], position_space)) - 1
    # return index of the nearest velocity state
    velocity_index = int(np.digitize(observation[1], velocity_space)) - 1
    # return the nearest discrete state
    observe = np.array([position_space[position_index], velocity_space[velocity_index]])
    return np.where((game_state == observe).all(axis= 1))[0][0]


start = time.time()
# Training phase
for i in range(episodes):
    init_state = env.reset()
    state = return_state(init_state)
    total_reward = 0
    for step in range(time_step):
        # uncomment env.render() to watch real time training in action
        #env.render()

        # epsilon greedy action
        if np.random.uniform(0,1) <= (1-epsilon):
            action = np.argmax(Q[state,:])
        else:
            action = np.random.randint(0,env.action_space.n)

        # epsilon decay
        step_count += 1
        epsilon = 1/np.sqrt(step_count) * epsilon

        new_observation, reward, done, info = env.step(action)
        new_state = return_state(new_observation)

        play_reward = reward
        # we give out positive reward when the car managed to reach the target point
        if done and step > -200:
            play_reward = 10

        # update Q table
        Q[state, action] = Q[state, action] + LearnRate*(play_reward + gamma*
                                                         np.max(Q[new_state,:]) - Q[state, action])
        total_reward += reward
        state = new_state

        if done:
            print("Running Episode: ", i+1, "Reward: ", total_reward)
            reward_history.append(total_reward)
            break

end = time.time()
print("Number of minutes took to train: ", (end-start)/60,"mins")

# plotting historical reward
# define helper to compute average reward over games
def moving_window(x, length):
    return [np.mean(x[i: i + length]) for i in range(0, (len(x)+1)-length)]

reward_history_plot_ = moving_window(reward_history, 25)
reward_history_plot_ = np.asarray(reward_history_plot_)
plt.plot(reward_history, color = "aquamarine", label = "Actual Reward")
plt.plot(reward_history_plot_, color = "teal", label = "Average Reward")
plt.xlabel("Episodes")
plt.ylabel("Episode Rewards (Average)")
plt.title("Episode Rewards over Time (Avg. over window size = 25)")
plt.legend()

