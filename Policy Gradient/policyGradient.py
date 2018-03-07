import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
env._max_episode_steps = 10000
env.reset()

#Set hyperparameters
H = 10 ##hidden layer neurons
batch_size = 5 #updating network every 5 episodes
LearnRate = 0.01
gamma = 0.95
input_dim = env.observation_space.shape[0]

#Initialising network with tensor
"""
We attempt to build neural network for this policy.
W1 = RELU, W2- score 
(However, it is possible to use others like gaussian, softmax, etc.. for this
policy network)
~~input a state and the network will output the policy
"""
tf.reset_default_graph()
input = tf.placeholder(tf.float32, [None, input_dim], name= "x_input")
W1 = tf.get_variable(shape= [input_dim, H], initializer= tf.contrib.layers.xavier_initializer(),
                     name= "W1")
hidden_layer = tf.nn.relu(tf.matmul(input, W1))
W2 = tf.get_variable( shape= [H,1], initializer= tf.contrib.layers.xavier_initializer(),
                      name= "W2")
score = tf.matmul(hidden_layer, W2)
prob = tf.nn.sigmoid(score)


train_vars = tf.trainable_variables()
y_input = tf.placeholder(tf.float32, [None, 1], name= "y_input")
advantage = tf.placeholder(tf.float32, name= "reward_signal")

#define loss function
loglik = tf.log(y_input*(y_input - prob) + (1 - y_input)*(y_input + prob))
loss = -tf.reduce_mean(loglik * advantage)
newGrad = tf.gradients(loss, train_vars)

#optimiser
adam = tf.train.AdamOptimizer(learning_rate= LearnRate)
W1Gradient = tf.placeholder(tf.float32, name= "batch_grad1")
W2Gradient = tf.placeholder(tf.float32, name= "batch_grad2")
batchGrad = [W1Gradient, W2Gradient]
updateGrads = adam.apply_gradients(zip(batchGrad, train_vars))

#def discount rewards
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(np.arange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

#set list to store items before update
xs, hs, ys, drs = [], [], [], []

total_episodes = 4000
reward_sum = 0
time_step = 350
PGreward_history = []
tf.set_random_seed(123)
np.random.seed(123)
initialise = tf.initialize_all_variables()

#start training
with tf.Session() as sess:
    sess.run(initialise)

    #store gradient for each interval step
    GradBuffer = sess.run(train_vars)
    for i, grad in enumerate(GradBuffer):
        GradBuffer[i] = grad * 0

    for epi in range(total_episodes):
        #reset for every new episodes
        state = env.reset()
        total_reward = 0
        xs, hs, ys, drs = [], [], [], []

        for _ in range(time_step):
            x_input = state.reshape([1, input_dim])
            action_prob = sess.run(prob, feed_dict= {input: x_input})
            #generate action
            if np.random.rand() < action_prob:
                action = 1
            else:
                action = 0
            xs.append(x_input)
            #implement fake label
            if action == 0:
                y = 1
            else:
                y = 0
            ys.append(y)

            #play action
            state, reward, done, info = env.step(action)
            total_reward += reward
            #store reward for every prev action made
            drs.append(reward)

            if done:
                # Stack the memory arrays to feed in session
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)

                # Compute the discounted reward
                discounted_epr = discount_rewards(epr)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                # Gen and save the gradient
                gradient = sess.run(newGrad, feed_dict={input: epx, y_input: epy, advantage: discounted_epr})
                for ix, grad in enumerate(gradient):
                    GradBuffer[ix] += grad

                # update parameters
                if (epi + 1) % batch_size == 0:
                    sess.run(updateGrads, feed_dict={W1Gradient: GradBuffer[0], W2Gradient: GradBuffer[1]})

                    #reset buffer after every updates
                    for ix, grad in enumerate(GradBuffer):
                        GradBuffer[ix] = grad * 0
                break

        print("Running Episode: ", epi + 1, "Reward: ", total_reward)
        PGreward_history.append(total_reward)













