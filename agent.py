import numpy as np
import tensorflow as tf
import pdb
import random
from tensorflow import keras
from tensorflow.keras import layers
from utilfunctions import one_hot
from utilfunctions import scale_state



class agent_learner:
    '''
    the agent class which has the Q nets:
    one is the one whihc is learned and the other one is the target one
    The Q nets have similar structures. The input for the Qnetwork is
    (state, action) pair and the output is the Q value for the input.
    The dimesion of the input (None, nr_features+nr_actions) and the
    outputs are of shape (None, 1), where None is the batchsize dimension
    '''
    
    def __init__(self, nr_features, nr_actions, gamma=0.99,  stddev=0.2, learning_rate=0.01):
        '''
        initializes the Q nets
        '''
        # the Q network
        initializer_Q = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev, seed=1)
        identity_Q = tf.keras.initializers.Identity()
        optimizer_Q = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        inputs_Q = keras.layers.Input(shape=(nr_features + nr_actions), name='state_and_action')
        x_Q = layers.Dense(6, activation='linear', kernel_initializer=identity_Q, name='linear_dense_Q')(inputs_Q)
        x_Q = layers.Dense(256, activation='relu', kernel_initializer=initializer_Q, name='relu_dense_Q_1')(x_Q)
        x_Q = layers.Dense(128, activation='relu', kernel_initializer=initializer_Q, name='relu_dense_Q_2')(x_Q)
        x_Q = layers.Dense(64, activation='relu', kernel_initializer=initializer_Q,  name='relu_dense_Q_3')(x_Q)
        output_Q = layers.Dense(1, activation='relu', kernel_initializer=initializer_Q, name='Q_value')(x_Q)
        self.Q = keras.Model(inputs=inputs_Q, outputs=output_Q)
        self.Q.compile(optimizer=optimizer_Q, loss=['mse'])

        # the target Q-network (the prime one)
        initializer_Q_t = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev, seed=1)
        identity_Q_t = tf.keras.initializers.Identity()
        optimizer_Q_t = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        inputs_Q_t = keras.layers.Input(shape=(nr_features + nr_actions), name='state_and_action')
        x_Q_t = layers.Dense(6, activation='linear', kernel_initializer=identity_Q_t, name='linear_dense_Qt')(inputs_Q_t)
        x_Q_t = layers.Dense(256, activation='relu', kernel_initializer=initializer_Q_t,  name='relu_dense_Qt_1')(x_Q_t)
        x_Q_t = layers.Dense(128, activation='relu', kernel_initializer=initializer_Q_t,  name='relu_dense_Qt_2')(x_Q_t)
        x_Q_t = layers.Dense(64, activation='relu', kernel_initializer=initializer_Q_t, name='relu_dense_Qt_3')(x_Q_t)
        output_Q_t = layers.Dense(1, activation='relu', kernel_initializer=initializer_Q_t, name='Q_target_value')(x_Q_t)
        self.Q_t = keras.Model(inputs=inputs_Q_t, outputs=output_Q_t)
        self.Q_t.compile(optimizer=optimizer_Q_t, loss=['mse'])

        self.gamma = gamma


    def prepare_learning_materials(self, events, env):
        '''
        Creating the y vector for learning. 
        The y vector is 
        y(s,a) := r(s,a) + gamma * Q_t(s', argmax_a'(Q(s',a'))
        with  Q_t -- the target net
        

        Keyword arguments:
        events -- a list of events
        env -- the environment

        returns:
        y vector
        '''
        debug = False #True

        nr_samples = len(events)

        s_primes = [x.scaled_state_prime for x in events]
        s_primes = np.array(s_primes)
        s_primes = np.reshape(s_primes, (nr_samples, -1))

        r = [x.reward for x in events] 
        r = np.array(r)
        r = np.reshape(r, (nr_samples, 1))

        done = [x.done for x in events]
        done = np.array(done)
        done = np.reshape(done, (nr_samples, 1))

        nr_actions = env.action_space.n

        if (debug):
            pdb.set_trace()

        for action_id in range(nr_actions):
            action = one_hot(action_id, nr_actions=env.action_space.n)
            actions = np.full((nr_samples, nr_actions), action)
            inputs_for_Q = np.concatenate((s_primes, actions), axis=1)
            if action_id == 0:
                tmp = self.Q.predict(inputs_for_Q)
            else:
                tmp = np.concatenate((tmp, self.Q.predict(inputs_for_Q)), axis=1)
                
        tmp = np.argmax(tmp, axis=1)
        if (debug):
            pdb.set_trace()
        inputs_for_Q_t = np.concatenate((s_primes, tf.one_hot(tmp, depth=nr_actions)), axis=1)
        y = r + self.gamma * self.Q_t.predict(inputs_for_Q_t) * (1 - done)
        return y
            

    def learn(self, events, env):
        '''
        fits the Q using events:
        
        1- creates the y vector

        2- creates the X vector which is made of the state, action pairs

        3- fits the Q network using X, y
        '''

        # 1
        y = self.prepare_learning_materials(events, env)

        # 2
        nr_samples = len(events)
        s = [x.scaled_state for x in events]
        s = np.array(s)
        s = np.reshape(s, (nr_samples, -1))
        actions = [x.action for x in events]
        actions = np.reshape(actions, (nr_samples, -1))
        X = np.concatenate((s, actions), axis=1)
        my_callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        ]
        
        # 3
        self.Q.fit(X, y, epochs=1, verbose=0, callbacks=my_callbacks)

    def update_Q_t_to_Q(self):
        '''
        set the weights of Q_t to the weights of Q
        '''
        self.Q_t.set_weights(self.Q.get_weights())

    def action_based_on_Q_target(self, state, env, epsilon):
        '''
        takes an action based on the epsilon greedy policy using the Q-target
        1 - for each state checks the predicted Q values for all the actions
        2 - pick the largest Q value
        3 - pick an action based on the largest Q value and epsilon
        
        Keyword arguments:
        
        state -- current state
        env -- the environment
        epsilon -- the epsilon in epsilon greedy approach

        returns:

        the id of the chosen action
        '''
        
        debug = False
        nr_samples = 1
        nr_actions = env.action_space.n
        nr_features = env.observation_space.high.shape[0]
        scaled_state = scale_state(state, env)
        scaled_state = np.array(scaled_state)
        scaled_state = np.reshape(scaled_state, (nr_samples, -1))
        if debug:
            print("scaled_state", scaled_state)
            pdb.set_trace()

        for action_id in range(nr_actions):
            action = one_hot(action_id, nr_actions=env.action_space.n)
            #actions = np.full((nr_samples, nr_actions), action)
            #import pdb; pdb.set_trace()
            inputs_for_Q_t = np.concatenate((scaled_state, action), axis=1)
            if action_id == 0:
                tmp = self.Q_t.predict(inputs_for_Q_t)
                if debug:
                    print("the predicted Q for action", action_id, " is ", tmp)
            else:
                tmp = np.concatenate((tmp, self.Q_t.predict(inputs_for_Q_t)), axis=1)
                if debug:
                    print("the predicted Q for action", action_id, " is ", tmp)
        tmp = tmp[0]
#        if (np.max(tmp) != 0):
#            tmp = tmp / (np.max(tmp))
        probabilities = tf.math.softmax(tmp)
        probabilities = (probabilities + epsilon) / (1.0 + epsilon * nr_actions)
        probabilities = probabilities.numpy()
        probabilities = probabilities / np.sum(probabilities)
        if debug:
            print(probabilities, np.sum(probabilities)-1)
        chosen_act = np.random.choice(nr_actions, p=probabilities)
        if debug:
            print("chosen_act", chosen_act)
            pdb.set_trace()

        return chosen_act
