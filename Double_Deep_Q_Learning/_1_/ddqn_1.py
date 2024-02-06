from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np


class DDQNReplayBuffer(object):

    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.memory_max_size = max_size
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.discrete = discrete
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.memory_max_size, self.input_shape))
        self.new_state_memory = np.zeros((self.memory_max_size, self.input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.memory_max_size, self.n_actions),
                                      dtype=dtype)
        self.reward_memory = np.zeros(self.memory_max_size)
        self.terminal_memory = np.zeros(self.memory_max_size, dtype=np.float32)

    def store_transition(self, state, state_, action, reward, done):
        index = self.mem_cntr % self.memory_max_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.memory_max_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        done = self.terminal_memory[batch]

        return states, actions, rewards, new_states, done


class DDQNModel(Model):

    def __init__(self, input_observation_dims, n_actions):
        super(DDQNModel, self).__init__()

        self.d1 = Dense(256, input_shape=(input_observation_dims,))
        self.activ1 = Activation('relu')

        self.d2 = Dense(256)
        self.activ2 = Activation('relu')

        self.d3 = Dense(256)
        self.activ3 = Activation('relu')

        self.fd = Dense(n_actions)

        # A small simple model should work for now.
        # I can always scale it up later

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.activ1(x)

        x = self.d2(x)
        x = self.activ2(x)

        x = self.d3(x)
        x = self.activ3(x)

        x = self.fd(x)

        return x


def create_dqn(input_observation_dims, learning_rate, n_actions):
    model = DDQNModel(input_observation_dims=input_observation_dims,
                      n_actions=n_actions)
    optimizers = Adam(learning_rate)
    model.compile(optimizer=optimizers, loss='mse')

    return model


class DDQNAgent(object):

    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_decay=0.0996, epsilon_end=0.01,
                 max_memory_size=1000000, fname='ddqn_model', replace_target=100):
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.max_memory_size = max_memory_size
        self.fname = fname
        self.replace_target = replace_target
        self.action_space = [i for i in range(self.n_actions)]
        self.memory = DDQNReplayBuffer(max_size=self.max_memory_size,
                                       input_shape=self.input_dims,
                                       n_actions=n_actions, discrete=True)
        self.q_eval = create_dqn(input_observation_dims=input_dims,
                                 n_actions=n_actions, learning_rate=self.alpha)
        self.q_target = create_dqn(input_observation_dims=input_dims,
                                   n_actions=n_actions, learning_rate=self.alpha)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(action=action, state=state, state_=new_state,
                                     reward=reward, done=done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        random = np.random.random()
        if random < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)

            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma * q_next[
                batch_index, max_actions.astype(int)] * done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end

            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        self.q_eval.save(self.fname)

    def load_model(self):
        self.q_eval = load_model(self.fname)

        if self.epsilon <= self.epsilon_end:
            self.update_network_parameters()

