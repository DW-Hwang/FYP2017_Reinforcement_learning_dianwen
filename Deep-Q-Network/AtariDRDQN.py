import gym
import atari_py
import numpy as np
from collections import deque
import keras
from keras.models import Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Input, Flatten, Conv2D, LSTM, TimeDistributed, Reshape, Permute
from keras.optimizers import Adam
from scipy.misc import imresize
import time
import os as os
import pandas as pd
import matplotlib.pyplot as plt

###############################################
#####   Double Recurrent Deep-Q-Nerwork   #####
###############################################

# Deterministic implements the frame skipping 
env = gym.make("BreakoutDeterministic-v4")
env._max_episode_steps = 100000
env.reset()

# uncheck if you are using GPU to run
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.30)
# sess = tf.Session(config= tf.ConfigProto(gpu_options= gpu_options))
# set_session(sess)

class double_recurrentDQN:
    def __init__(self, env, batch_size, epsilon_decay, epsilon, gamma):
        self.env = env
        self.memory = Replay_Buffer()
        self.reward_history = []
        self.loss_history = []
        self.minEpsilon = 0.1
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.gamma = gamma
        self.step_count = 0
        self.Qtarget = self.BuildModel()
        self.QModel = self.BuildModel()

    # Pre-process observed state(image) for faster computation
    def preprocIMG(self, frame):
        frame = frame[35:195]  # crop vertically, removing score board
        frame = frame[::2, ::2, 0]  # greyscale
        frame[frame == 144] = 0  # remove background
        frame[frame == 109] = 0  # remove background
        frame[frame != 0] = 1  # Set 1 to the Rest (paddle, ball)
        frame = imresize(frame, size=(84, 84), interp='nearest') # Resize to 84,84,1
        return frame

    # Double Recurrent Deep Q Network
    def BuildModel(self):
        input_layer = Input(shape= (84,84,4))
        input_data_expanded = Reshape((84, 84, 4, 1), input_shape=(84,84,4))(input_layer)
        input_data_TimeDistributed = Permute((3, 1, 2, 4), input_shape=(84,84,4))(input_data_expanded)
        conv1 = TimeDistributed(Conv2D(32, 8, strides=(4,4), activation= "relu"))(input_data_TimeDistributed)
        conv2 = TimeDistributed(Conv2D(64, 4, strides= (2,2), activation= "relu"))(conv1)
        conv3 = TimeDistributed(Conv2D(64, 3, strides= (1,1), activation= "relu"))(conv2)
        flatten = TimeDistributed(Flatten())(conv3)
        fully_connected_layer = TimeDistributed(Dense(512, activation= "relu"))(flatten)
        LSTM_layer = LSTM(512, return_sequences= False,stateful=False, input_shape=(4, 512))(fully_connected_layer)
        output = Dense(self.env.action_space.n)(LSTM_layer)
        model = Model(inputs= [input_layer], outputs= [output])
        model.compile(optimizer= Adam(lr=0.001), loss="mse")
        return model

    def update_target(self):
        weights = self.QModel.get_weights()
        self.Qtarget.set_weights(weights)

    def choose_action(self, state):
        # if prob > epsilon, we generate action from model
        if np.random.uniform(0,1) <= 1 - self.epsilon:
            state = np.dstack(state)[None,]
            return np.argmax(self.QModel.predict(state)) # this generate action from model
        else:
            # we explore and generate random moves
            return np.random.randint(0,self.env.action_space.n)

    def epsilon_update(self):
        self.epsilon = self.minEpsilon + \
                       0.90*(np.exp(-self.epsilon_decay * self.step_count))

    def train_agent(self, episodes, time_step):

        for epi in range(episodes):
            #initialise stateList- shape: (4,84,84)
            stateList = [self.preprocIMG(self.env.reset())] * 4
            total_reward = 0
            ale_lives = 5
            loss = 0.

            #collect replay experience
            for _ in range(time_step):
                # uncomment "self.env.render()" to watch agent's live play
                #self.env.render()
                action = self.choose_action(stateList)
                next_state, reward, done, info = self.env.step(action)
                next_stateList = stateList[1:]
                next_stateList.append(self.preprocIMG(next_state))

                '''we change the reward system in this env'''
                # reward = -1 :failing to catch the ball and losing a live
                # reward = +1 : hitting some bricks
                agent_reward = reward
                if info["ale.lives"] < ale_lives:
                    agent_reward = -1
                    ale_lives = info["ale.lives"]

                self.memory.addMemory((stateList, action, agent_reward, next_stateList, done))
                total_reward += reward
                stateList = next_stateList
                self.step_count += 1

                if done:
                    break

            # Training phase
            # we only start training after collecting 20000 frames
            if len(self.memory.getMemory()) >= self.batch_size:
                minibatch = self.memory.getMiniBatch(self.batch_size)
                x_input = []
                y_output = []
                for experience in minibatch:
                    stateList = np.dstack(experience[0])
                    action = experience[1]
                    reward = experience[2]
                    next_stateList = np.dstack(experience[3])
                    done = experience[4]

                    # updating our network
                    x_input.append(stateList)
                    target = list(self.QModel.predict(stateList[None,])[0])
                    if not done and reward >= 0:
                        Qtarget_ = list(self.Qtarget.predict(next_stateList[None,])[0])
                        action_next = np.argmax(self.QModel.predict(next_stateList[None,]))
                        target[action] = reward + self.gamma * Qtarget_[action_next]
                        y_output.append(target)
                    else:
                        target[action] = reward
                        y_output.append(target)
                loss = self.QModel.train_on_batch(np.array(x_input), np.array(y_output))
                self.loss_history.append(loss)

            # Updating our target model with small delay.
            if self.step_count % 30 == 0:
                self.update_target()

            # decay epsilon
            self.epsilon_update()

            print("Running Episode: ", epi + 1, ", Scores: ", total_reward, ", loss: ", loss)
            self.reward_history.append(total_reward)

    def play_agent(self, episodes, time_step):
        self.playscore = []
        for i in range(episodes):
            #initialise state
            stateList = [self.preprocIMG(self.env.reset())] * 4
            total_reward = 0
            for _ in range(time_step):
                self.env.render()
                action = self.choose_action(stateList)
                state, reward, done, info = self.env.step(action)
                stateList.pop(0)
                stateList.append(self.preprocIMG(state))
                total_reward += reward
                if done:
                    break
            print("Running Episode: ", i + 1, "Reward: ", total_reward)
            self.playscore.append(total_reward)
        self.env.render(close=True)


''' Building our replay buffer to store the experience of each game '''

class Replay_Buffer:
    def __init__(self):
        self.buffer = deque(maxlen= 100000)

    def getMemory(self):
        return self.buffer

    def addMemory(self, experience):
        self.buffer.append(experience)

    def getMiniBatch(self, batch_size):
        output = []
        index = np.random.choice(np.arange(8, len(self.buffer)),
                                 size=batch_size, replace=False)
        for item in index: 
            for i in np.arange(item-8,item):
                output.append(self.buffer[i])
        return output



# Set hyperparameters
train_episode = 30000
time_steps = 100000
batch_size = 32
epsilon_decay = 0.000685
epsilon = 1
gamma = 0.99
play_episode = 50
np.random.seed(123)


"""Train Session"""
Breakout008 = double_recurrentDQN(env, batch_size, epsilon_decay, epsilon, gamma)
start = time.time()
Breakout008.train_agent(train_episode, time_steps)
end = time.time()
print("Hours took to train: ", (end-start)/3600)

# saving trained model
Breakout008.QModel.save("DRDQN_model.h5")
Breakout008.play_agent(play_episode, time_steps)
