import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from PIL import Image
import time
import ale_py


# Même architecture que celle utilisée pour l'entraînement
class DQNModel(nn.Module):
    def __init__(self, input_shape, nb_actions):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, nb_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # aplatissement
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Traitement de l'image pour correspondre à l'entrée du modèle
class ImageProcessor:
    def __init__(self):
        self.IMG_SHAPE = (84, 84)

    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(self.IMG_SHAPE).convert('L')
        img = np.array(img)
        return img.astype('uint8')

# Fonction principale pour charger le modèle et exécuter une partie en mode test avec interface visuelle
def main():
    # Hyperparamètres
    WINDOW_LENGTH = 12
    nb_actions = 6
    IMG_SHAPE = (84, 84)
    input_shape = (WINDOW_LENGTH,) + IMG_SHAPE

    # Création de l'environnement avec affichage (mode "human")
    env = gym.make('ALE/Pong-v5', render_mode="human")
    processor = ImageProcessor()

    # Création du modèle et chargement des poids
    model = DQNModel(input_shape, nb_actions)
    checkpoint = torch.load("checkpoints/current_best.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # Initialisation de l'état : on empile WINDOW_LENGTH images traitées
    state, info = env.reset()
    processed = processor.process_observation(state)
    state = np.stack([processed] * WINDOW_LENGTH, axis=0)

    done = False
    episode_reward = 0
    while not done:
        # Affichage de l'environnement (le mode "human" s'en charge automatiquement)
        # On choisit une action de manière entièrement déterministe (greedy)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # ajouter une dimension batch
        with torch.no_grad():
            q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Traitement de la nouvelle observation et mise à jour de l'état
        processed_next = processor.process_observation(next_state)
        next_state_stack = np.append(state[1:], [processed_next], axis=0)

        done = terminated or truncated
        state = next_state_stack

        # Petite pause pour laisser le temps au rendu d'être affiché
        time.sleep(0.004)

    print("Episode terminé avec un score de:", episode_reward)
    env.close()

if __name__ == "__main__":
    main()
