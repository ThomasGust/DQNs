from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np


class ReplayBuffer(object):

    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.memory_max_size = max_size
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.discrete = discrete
        self.state_memory = np.zeros((self.memory_max_size, self.input_shape))
        self.new_state_memory = np.zeros((self.memory_max_size, self.input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.memory_max_size, self.n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.memory_max_size)
        self.terminal_memory = np.zeros(self.memory_max_size, dtype=np.float32)
        self.mem_cntr = 0

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.memory_max_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.memory_max_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class SimpleDQN1(Model):

    def __init__(self, n_actions, input_dims,
                 d1d, d2d, d3d, d4d, d5d):
        super(SimpleDQN1, self).__init__()

        self.d1 = Dense(d1d, input_shape=(input_dims,))
        self.activ1 = Activation('relu')

        self.d2 = Dense(d2d)
        self.activ2 = Activation('relu')

        self.d3 = Dense(d3d)
        self.activ3 = Activation('relu')

        self.d4 = Dense(d4d)
        self.activ4 = Activation('relu')

        self.d5 = Dense(d5d)
        self.activ5 = Activation('relu')

        self.fd = Dense(n_actions)

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.activ1(x)

        x = self.d2(x)
        x = self.activ2(x)

        x = self.d3(x)
        x = self.activ3(x)

        x = self.d4(x)
        x = self.activ4(x)

        x = self.d5(x)
        x = self.activ5(x)

        x = self.fd(x)

        return x


def build_model(lr, n_actions, input_dims,
                d1d, d2d, d3d, d4d, d5d):
    model = SimpleDQN1(n_actions=n_actions,
                       input_dims=input_dims,
                       d1d=d1d, d2d=d2d,
                       d3d=d3d, d4d=d4d,
                       d5d=d5d)
    optimizer = Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='mse')

    return model


class SimpleDQNAgent(object):

    def __init__(self, learning_rate, discount_factor, n_actions, epsilon, batch_size,
                 input_dims, epsilon_decay=0.996, epsilon_end=0.01,
                 mem_size=1000000, fname='simple_dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.mem_size = mem_size
        self.fname = fname

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)

        self.q_eval = build_model(learning_rate, n_actions, input_dims,
                                  1000, 1000, 1000, 1000, 1000)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.discount_factor*np.max(q_next, axis=1) * done

        _ = self.q_eval.fit(state, q_target, verbose=0)

        self.epsilon = self.epsilon*self.epsilon_decay if self.epsilon > \
            self.epsilon_end else self.epsilon_end

    def save_model(self):
        self.q_eval.save(self.fname)

    def load_model(self):
        self.q_eval = load_model(self.fname)