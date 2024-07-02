import cv2
import numpy as np
import pyautogui
import time
import os
import math

from learn import train_model as tm
from mouse import HandTracking as htm
import mediapipe as mp
from sklearn.preprocessing import MinMaxScaler
from pynput.mouse import Controller, Button

import torch
import torch.nn as nn
import torch.nn.functional as F

mouse = Controller()
# 볼륨 설정 함수
def set_volume(volume_percent):
    os.system(f"osascript -e 'set volume output volume {volume_percent}'")

before_brightness = 0
# 밝기 조절 함수
def set_brightness(brightness_percent):
    global before_brightness
    # os.system(f"brightness {brightness_percent}") => 안됨
    if before_brightness < brightness_percent:
        os.system("osascript -e 'tell application \"System Events\"' -e 'key code 144' -e 'end tell'")
        before_brightness = brightness_percent
    else:
        os.system("osascript -e 'tell application \"System Events\"' -e 'key code 145' -e 'end tell'")
        before_brightness = brightness_percent


# 제스처 탐지 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
gesture_folders = {'none': 0, 'rock': 1, 'paper': 2, 'scissors': 3, 'temp': 4, "duck": 5,'twist': 6, '1k': 7, "2k": 8}
gesture_folders_small = {
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
    'four' : 6

}
model = tm.ImageModel()
model.load_state_dict(torch.load("final_model.pth"))
model.eval()
#
# model = tm.trained_model()
# model.eval()

# 마우스 설정
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)
cap.set(cv2.CAP_PROP_FPS, 30)
detector = htm.handDetector(maxHands=1)

wScr, hScr = pyautogui.size()

mouse_control_enabled = False

recognize_mode = True

control_mode = True

# 타이머 설정
last_action_time = 0
action_cooldown = 2  # 지연 시간을 1초로 설정

last_click_time = 0
click_cooldown = 1

order_mode = True
def calculate_feature(hand_landmarks, image_width, image_height):
    # joint = np.zeros((21, 2))
    # for j, lm in enumerate(hand_landmarks.landmark):
    #     joint[j] = [lm.x * image_width, lm.y * image_height]
    #
    joint = np.zeros((21, 3))
    for j, lm in enumerate(hand_landmarks.landmark):
        joint[j] = [lm.x * image_width, lm.y * image_height,lm.z]


    # 벡터 계산
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
    v = v2 - v1  # 벡터
    v_mag = np.linalg.norm(v, axis=1)[:, np.newaxis]

    v = v / v_mag  # 단위벡터

    # 각도 계산
    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # 내적
    angle = np.degrees(angle)  # 라디안을 도로 변환

    scaler = MinMaxScaler()
    scaled_angles = scaler.fit_transform(angle.reshape(-1, 1))
    scaled_mag = scaler.fit_transform(v_mag.reshape(-1, 1))

    feature = np.concatenate((scaled_angles, scaled_mag)).flatten()

    return angle

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        current_time = time.time()
        success, img = cap.read()
        if success:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks and recognize_mode:
                for hand_landmarks in results.multi_hand_landmarks:
                    angles = calculate_feature(hand_landmarks, img.shape[1], img.shape[0])
                    X_pred = np.array([angles], dtype=np.float32)

                    with torch.no_grad():  # 기울기 계산을 비활성화
                        output = model(torch.tensor(X_pred, dtype=torch.float32))
                        probabilities = F.softmax(output, dim=1)  # 결과를 확률로 변환
                        max_probs, predictions = torch.max(probabilities, 1)  # 최대 확률과 해당 인덱스
                        predictions[max_probs < 0.8] = 0  # 임계값 미만인 경우 0번 레이블로 변경

                    gesture_id = predictions.item()
                    gesture_name = [name for name, idx in gesture_folders_three.items() if idx == gesture_id][0]

                    print(gesture_name)
                    if current_time - last_action_time > action_cooldown:
                        if gesture_name == "paper":
                            # # 볼륨 조절 모드 or 디스플레이 밝기 조절 전환
                            # control_mode = not control_mode
                            # if control_mode:
                            #     print("sound control mode !!")
                            #     cv2.putText(img, 'sound control', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
                            # else:
                            #     print("display brightness control mode !!")
                            #     cv2.putText(img, 'display brightness control', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            #                 (255, 255, 255), 2, c v2.LINE_AA)
                            recognize_mode = False
                            # mouse_control_enabled = True
                            last_action_time = current_time  # 마지막 동작 시간 업데이트

                        elif gesture_name == "one":
                            if order_mode:
                                pyautogui.hotkey('j')
                                cv2.putText(img, 'Rewind 10 seconds', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                            cv2.LINE_AA)
                            else:
                                pyautogui.hotkey('k')
                                cv2.putText(img, 'Toggle play/pause', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 255), 2, cv2.LINE_AA)
                            last_action_time = current_time

                        elif gesture_name == "scissors":
                            if order_mode:
                                pyautogui.hotkey('l')
                                cv2.putText(img, 'Skip forward 10 seconds', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                            cv2.LINE_AA)
                            else:
                                pyautogui.hotkey('f')
                                cv2.putText(img, 'Toggle fullscreen', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                            cv2.LINE_AA)
                            last_action_time = current_time

                        elif gesture_name == "ok":
                            cv2.putText(img, 'command change', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                            if order_mode:
                                order_mode = False
                            else:
                                order_mode = True
                            last_action_time = current_time


                        elif gesture_name == "three":
                            if order_mode:
                                pyautogui.hotkey('shift', '.')
                                cv2.putText(img, 'Speed up playback', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                            cv2.LINE_AA)
                            else:
                                pyautogui.hotkey('m')
                                cv2.putText(img, 'Mute', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                            2, cv2.LINE_AA)
                            last_action_time = current_time



                        elif gesture_name == "four":
                            if order_mode:
                                pyautogui.hotkey('shift', ',')
                                cv2.putText(img, 'Speed slow down playback', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 255), 2,
                                            cv2.LINE_AA)
                            else:
                                pyautogui.hotkey('c')
                                cv2.putText(img, 'Subtitles', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                            2, cv2.LINE_AA)
                            last_action_time = current_time
                    # mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img)
            fingers = detector.fingersUp()

            # if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            #     recognize_mode = False

            if not recognize_mode:
                if not mouse_control_enabled:
                    # 여기에서 마우스의 초기 위치를 설정
                    plocX, plocY = pyautogui.position()
                    mouse_control_enabled = True
                current_mouse_time = time.time()
                if len(lmList) != 0:
                    x1, y1 = lmList[8][1:]
                    x2, y2 = lmList[12][1:]
                    fingers = detector.fingersUp()

                    # 마우스 커서 이동 코드
                    if fingers[1] == 1 and fingers[2] == 0:
                        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening
                        # pyautogui.moveTo((wScr - clocX, clocY))
                        mouse.position = (wScr - clocX, clocY)

                        # Display the coordinates on the image
                        # cv2.putText(img, f'position : ({int(clocX)}, {int(clocY)})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        #             1, (255, 255, 255), 2, cv2.LINE_AA)

                        # cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                    # 볼륨 조절
                    if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                        print("조절 모드")
                        if control_mode:
                            x_thumb, y_thumb = lmList[4][1], lmList[4][2]  # 엄지손가락 좌표
                            x_index, y_index = lmList[8][1], lmList[8][2]  # 검지손가락 좌표
                            distance = math.hypot(x_index - x_thumb, y_index - y_thumb)  # 두 손가락 사이의 유클리디안 거리 계산
                            print("distance: ",distance)
                            # 거리에 따라 볼륨 조절 (예를 들어, 거리가 30px에서 200px 사이라고 가정)
                            vol = np.interp(distance, [20, 120], [0, 100])
                            print("vol: ", vol)
                            set_volume(vol)  # 볼륨 설정 함수 호출

                            # 볼륨 상태를 이미지에 표시
                            # cv2.putText(img, f'length : ({int(distance)})', (50, 50),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
                            volBar = np.interp(vol, [0, 100], [400, 150])
                            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                            cv2.putText(img, f'{int(vol)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
                        else:
                            x_thumb, y_thumb = lmList[4][1], lmList[4][2]  # 엄지손가락 좌표
                            x_index, y_index = lmList[8][1], lmList[8][2]  # 검지손가락 좌표
                            distance = math.hypot(x_index - x_thumb, y_index - y_thumb)  # 두 손가락 사이의 유클리디안 거리 계산

                            # 거리를 밝기 수준으로 매핑 (예를 들어, 거리가 30px에서 200px 사이라고 가정)
                            brightness = np.interp(distance, [30, 200], [0, 100])  # 밝기를 0.1에서 1 사이로 조절

                            # 밝기 조절 함수 호출
                            set_brightness(int(brightness))

                            # 밝기 상태를 이미지에 표시
                            # cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
                            # brightBar = np.interp(brightness, [0, 100], [400, 150])
                            # cv2.rectangle(img, (50, int(brightBar)), (85, 400), (0, 0, 255), cv2.FILLED)
                            # cv2.putText(img, f'{int(brightness * 100)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                            #             (255, 255, 255), 3)

                    # 클릭 이벤트 처리 코드
                    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                        print("click 준비")
                        length, img, lineInfo = detector.findDistance(8, 12, img)
                        if length < 20:
                            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                            clocX = plocX + (x3 - plocX) / smoothening
                            clocY = plocY + (y3 - plocY) / smoothening
                            # pyautogui.mouseDown(wScr - clocX, clocY)
                            if current_mouse_time - last_click_time > click_cooldown:
                                mouse.click(Button.left, 1)
                                last_click_time = current_mouse_time
                            # cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                            plocX, plocY = clocX, clocY

                        # else:
                            # pyautogui.click()
                            # cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            # pyautogui.sleep(0.5)
                    # 이부분 다시 활성화 시켜야함
                    # if fingers[1] == 0:
                    #     mouse_control_enabled = False  # 마우스 컨트롤 비활성화
                    #     recognize_mode = True
                else:
                    mouse_control_enabled = False  # 마우스 컨트롤 비활성화
                    recognize_mode = True
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            # cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
