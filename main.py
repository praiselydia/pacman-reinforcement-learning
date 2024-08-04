import os
import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import cv2

# Create the MsPacman environment
env = gym.make('MsPacman-v4', render_mode="human")
state_size = (84, 84, 4)
action_size = env.action_space.n

# Hyperparameters
batch_size = 32
n_episodes = 50
output_dir = 'model_output/pacman/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

agent = DQNAgent(state_size, action_size)

# Resize and preprocess the game frames
def preprocess_frame(frame):
    # Debug: Print the shape and type of the frame
    print(f"Original frame shape: {frame.shape}, type: {type(frame)}")
    
    # Convert RGB to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Debug: Print the shape and type after converting to grayscale
    print(f"Grayscale frame shape: {gray.shape}, type: {type(gray)}")
    
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Debug: Print the shape and type after resizing
    print(f"Resized frame shape: {resized.shape}, type: {type(resized)}")
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    return np.reshape(normalized, (84, 84, 1))

done = False
for e in range(n_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # Extract the observation if it is a tuple
    state = preprocess_frame(state)
    state = np.stack([state] * 4, axis=3)  # Create a stack of 4 frames
    state = np.expand_dims(state, axis=0)  # Add batch dimension

    for time in range(5000):
        env.render()  # Call render without the mode argument
        action = agent.act(state)
        
        # Unpack the results from the environment step
        next_state, reward, done, truncated, info = env.step(action)
        print("Result from env.step(action):", (next_state, reward, done, truncated, info))  # Debugging line
        
        reward = reward if not done else -10
        if isinstance(next_state, tuple):
            next_state = next_state[0]  # Extract the observation if it is a tuple
        
        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=3)  # Add channel dimension
        next_state = np.expand_dims(next_state, axis=0)  # Add batch dimension

        # Remove batch dimension before concatenation
        state_no_batch = np.squeeze(state, axis=0)  # Remove batch dimension
        next_state_no_batch = np.squeeze(next_state, axis=0)  # Remove batch dimension

        # Concatenate along the channel axis
        state_no_batch = np.concatenate((state_no_batch[:, :, :, 1:], next_state_no_batch), axis=3)
        state = np.expand_dims(state_no_batch, axis=0)  # Add batch dimension back

        agent.remember(state, action, reward, next_state, done)
        state = np.expand_dims(state, axis=0)  # Ensure state has batch dimension
        if done or truncated:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if e % 10 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")

env.close()
