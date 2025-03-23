import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import random
from collections import deque
import numpy as np
from PIL import Image
import logging
import copy  # pour cloner le modèle

logging.basicConfig(level=logging.INFO, format="%(message)s")

class ImageProcessor:
    def process_observation(self, observation):
        IMG_SHAPE = (84, 84)
        img = Image.fromarray(observation)
        img = img.resize(IMG_SHAPE).convert('L')
        img = np.array(img)
        return img.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch / 255.0
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

class DQNModel(nn.Module):
    def __init__(self, input_shape, nb_actions):
        super(DQNModel, self).__init__()
        # Convolutions
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Fully connected layers
        # Note : la taille 7x7 est basée sur des dimensions d'entrée typiques (ex: 84x84)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, nb_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, model, optimizer, memory, nb_actions, target_update_interval=1000, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.model = model
        self.optimizer = optimizer
        self.memory = memory
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Création du target network, initialisé avec les mêmes poids que le modèle principal
        self.target_model = copy.deepcopy(self.model)
        self.target_update_interval = target_update_interval
        self.train_step = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.nb_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # ajouter une dimension batch
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def train(self, batch_size):
        if self.memory.size() < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Calcul des Q-values actuels pour les actions sélectionnées
        q_values = self.model(states).gather(1, actions).squeeze(1)

        # Calcul des Q-values cibles avec le target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mise à jour de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Mise à jour périodique du target network
        self.train_step += 1
        if self.train_step % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            logging.info(f"Target network mis à jour à l'étape {self.train_step}")

if __name__ == "__main__":
    # Hyperparamètres
    IMG_SHAPE = (84, 84)
    WINDOW_LENGTH = 12
    nb_actions = 6
    input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])
    learning_rate = 0.000005
    batch_size = 32
    replay_capacity = 1000000
    nb_steps = 1000000
    checkpoint_interval = 10000
    target_update_interval = 10000  # intervalle de mise à jour du target network

    env = gym.make('ALE/Pong-v5')

    model = DQNModel((WINDOW_LENGTH,) + IMG_SHAPE, nb_actions)
    checkpoint = torch.load("checkpoints/current_best.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Scheduler: diminue le learning rate de 10% toutes les 10 000 étapes
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    memory = ReplayBuffer(replay_capacity)
    processor = ImageProcessor()
    agent = DQNAgent(model, optimizer, memory, nb_actions, target_update_interval=target_update_interval)

    episode_rewards = []
    state, info = env.reset()
    state = np.stack([processor.process_observation(state)] * WINDOW_LENGTH, axis=0)

    episode_reward = 0
    # Phase de warm-up : collecte d'expériences aléatoires avant entraînement
    warmup_steps = 0
    logging.info(f"Début du warm-up pour {warmup_steps} steps...")
    print(f"Début du warm-up pour {warmup_steps} steps...")

    for step in range(warmup_steps):
        action = env.action_space.sample()  # Action aléatoire
        (next_state, reward, terminated, truncated, info) = env.step(action)
        next_state = processor.process_observation(next_state)
        next_state = np.append(state[1:], [next_state], axis=0)

        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)
        state = next_state

        if done:
            state, info = env.reset()
            state = np.stack([processor.process_observation(state)] * WINDOW_LENGTH, axis=0)

        if step % 10000 == 0:
            logging.info(f"Warm-up progress: {step}/{warmup_steps}")
            print(f"Warm-up progress: {step}/{warmup_steps}")

    logging.info("Warm-up terminé, début de l'entraînement...")
    print("Warm-up terminé, début de l'entraînement...")

    for step in range(nb_steps):
        action = agent.select_action(state)
        (next_state, reward, terminated, truncated, info) = env.step(action)
        next_state = processor.process_observation(next_state)
        next_state = np.append(state[1:], [next_state], axis=0)

        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)
        agent.train(batch_size)

        # Mise à jour du scheduler à chaque étape
        scheduler.step()

        state = next_state
        episode_reward += reward

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            state, info = env.reset()
            state = np.stack([processor.process_observation(state)] * WINDOW_LENGTH, axis=0)

        # Sauvegarde de checkpoint
        if step % checkpoint_interval == 0 and step > 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'episode_rewards': episode_rewards
            }, f"checkpoints/checkpoint_{step}.pth")
            logging.info(f"Checkpoint sauvegardé à l'étape {step}")
            print(f"Step {step} - Checkpoint sauvegardé.")

        if step % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            logging.info(f"Step {step}, Dernier avg reward sur 100 épisodes: {avg_reward:.2f}")
            print(f"Step {step}, Dernier avg reward sur 100 épisodes: {avg_reward:.2f}")

    env.close()

    # Sauvegarde finale du modèle
    torch.save(model.state_dict(), "dqn_pong_final.pth")
    logging.info("Entraînement terminé. Modèle sauvegardé sous dqn_pong_final.pth")
    print("Entraînement terminé. Modèle sauvegardé sous dqn_pong_final.pth")
