import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

gesture_folders = {
    'none': 0,
    'rock': 1,
    'paper': 2,
    'scissors': 3,
    'temp': 4,
    'duck': 5,
    'twist': 6,
    '1k': 7,
    "2k": 8
}

gesture_folders_thumb = {
    'none': 0,
    'one': 1,
    'paper': 2,
    'scissors': 3,
    'ok': 4,
    'thumb': 5,
}

gesture_folders_three = {
    'none': 0,
    'one': 1,
    'paper': 2,
    'scissors': 3,
    'ok': 4,
    'three': 5,
    'four': 6
}


def calculate_feature(hand_landmarks, image_width, image_height):
    joint = np.zeros((21, 3))
    for j, lm in enumerate(hand_landmarks.landmark):
        joint[j] = [lm.x * image_width, lm.y * image_height, lm.z]

    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
    v = v2 - v1

    v_mag = np.linalg.norm(v, axis=1)[:, np.newaxis]
    v = v / v_mag

    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
    angle = np.degrees(angle)

    scaler = MinMaxScaler()
    scaled_angles = scaler.fit_transform(angle.reshape(-1, 1))
    scaled_mag = scaler.fit_transform(v_mag.reshape(-1, 1))

    feature = np.concatenate((scaled_angles, scaled_mag)).flatten()

    return angle


def extract_features_and_labels():
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        features = []
        labels = []
        for gesture_name, label in gesture_folders_three.items():
            folder_path = os.path.join('train_data2', gesture_name)
            print(folder_path)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG')):
                    img_path = os.path.join(folder_path, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            angles = calculate_feature(hand_landmarks, img.shape[1], img.shape[0])
                            if angles is not None:
                                features.append(angles)
                                labels.append(label)
        return np.array(features), np.array(labels)


class ImageModel(nn.Module):
    def __init__(self, input_features=15, hidden_units=64, output_features=7):
        super(ImageModel, self).__init__()
        self.fc1 = nn.Linear(input_features, 32)
        self.fc2 = nn.Linear(32, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units * 2)
        self.fc4 = nn.Linear(hidden_units * 2, hidden_units)
        self.fc5 = nn.Linear(hidden_units, 32)
        self.dropout5 = nn.Dropout(0.25)
        self.fc6 = nn.Linear(32, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        x = self.fc6(x)
        return x


def postprocess_outputs(outputs, threshold=0.5):
    probabilities = F.softmax(outputs, dim=1)
    max_probs, predictions = torch.max(probabilities, 1)
    return predictions


def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predicted = postprocess_outputs(output)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def get_all_predictions(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.tolist())
            all_targets.extend(target.tolist())
    return all_preds, all_targets


def trained_model():
    features, labels = extract_features_and_labels()

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = ImageModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 50
    train_losses = []
    train_accuracy = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        average_loss = running_loss / num_batches
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")

        train_losses.append(average_loss)
        train_accuracy.append(evaluate_model(model, test_loader))

    predictions, targets = get_all_predictions(model, test_loader)
    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(10, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=gesture_folders_three,
                yticklabels=gesture_folders_three, annot_kws={"size": 20})
    plt.xlabel('Predicted Labels', fontsize=18)
    plt.ylabel('True Labels', fontsize=18)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label='Training loss')
    plt.title('Training Loss', fontsize=20)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.title('Training Accuracy', fontsize=20)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy (%)', fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.show()

    torch.save(model.state_dict(), 'final_model.pth')
    print("Model saved as 'trained_model.pth'")

    return model