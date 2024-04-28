import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.model_selection import train_test_split

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

gesture_folders = {
    'none': 0,
    'rock': 1,
    'paper': 2,
    'scissors': 3
}

# def calculate_angles(hand_landmarks, image_width, image_height):
#     joint = np.zeros((21, 3))
#     for j, lm in enumerate(hand_landmarks.landmark):
#         joint[j] = [lm.x * image_width, lm.y * image_height, lm.z]
#
#     # 벡터 계산
#     v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
#     v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
#     v = v2 - v1  # 벡터
#     v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
#
#     # 각도 계산
#     angle = np.arccos(np.einsum('nt,nt->n',
#                                 v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
#                                 v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # 내적
#     angle = np.degrees(angle)  # 라디안을 도로 변환
#
#     return angle
def calculate_angles(hand_landmarks, image_width, image_height):
    joint = np.zeros((21, 3))
    for j, lm in enumerate(hand_landmarks.landmark):
        joint[j] = [lm.x * image_width, lm.y * image_height, lm.z]

    # 모든 랜드마크의 평균 좌표를 원점으로 사용
    origin = np.mean(joint, axis=0)

    # 원점에서 각 랜드마크까지의 벡터 계산
    vectors = joint - origin

    # L2 거리로 벡터 정규화
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

    # 각도 계산
    # 내적을 이용하여 각도 계산
    angles = np.arccos(np.einsum('nt,nt->n',
                                vectors[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :],
                                vectors[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]))  # 내적
    angles = np.degrees(angles)  # 라디안을 도로 변환

    return angles

# 이미지에서 특징(각도)와 레이블을 추출하는 함수
def extract_features_and_labels():
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        features = []
        labels = []
        for gesture_name, label in gesture_folders.items():
            folder_path = os.path.join('rps_data_sample', gesture_name)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_name)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            angles = calculate_angles(hand_landmarks, img.shape[1], img.shape[0])
                            if angles is not None:
                                features.append(angles)
                                labels.append(label)
        return np.array(features), np.array(labels)


def trained_model():
    # 특징과 레이블 추출
    features, labels = extract_features_and_labels()
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # KNN 모델 훈련
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn




