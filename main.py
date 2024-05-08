import cv2
import numpy as np
import pyautogui
import time
import os
import math

from learn import train_model as tm
from mouse import HandTracking as htm
import mediapipe as mp

# 볼륨 설정 함수
def set_volume(volume_percent):
    os.system(f"osascript -e 'set volume output volume {volume_percent}'")


# 제스처 탐지 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
gesture_folders = {'none': 0, 'rock': 1, 'paper': 2, 'scissors': 3}
model = tm.trained_model()

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

# 타이머 설정
last_action_time = 0
action_cooldown = 1  # 지연 시간을 1초로 설정
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

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        current_time = time.time()
        success, img = cap.read()
        if success:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks and recognize_mode:
                for hand_landmarks in results.multi_hand_landmarks:
                    angles = calculate_angles(hand_landmarks, img.shape[1], img.shape[0])
                    X_pred = np.array([angles], dtype=np.float32)
                    gesture_id = model.predict(X_pred)[0]
                    gesture_name = [name for name, idx in gesture_folders.items() if idx == gesture_id][0]

                    if current_time - last_action_time > action_cooldown:
                        if gesture_name == "paper":
                            # 플랫폼에 따라 단축키 실행
                            pyautogui.hotkey('command' if os.name == 'posix' else 'ctrl', 'm')
                            cv2.putText(img, '최소화', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                            last_action_time = current_time  # 마지막 동작 시간 업데이트

                        elif gesture_name == "rock":
                            pyautogui.hotkey('ctrl' if os.name == 'posix' else 'ctrl', 'right')
                            cv2.putText(img, '데스크탑 이동', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                            last_action_time = current_time

                        elif gesture_name == "scissors":
                            pyautogui.hotkey('ctrl' if os.name == 'posix' else 'ctrl', 'left')
                            cv2.putText(img, '데스크탑 이동', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                        cv2.LINE_AA)
                            last_action_time = current_time

                    # mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img)
            fingers = detector.fingersUp()

            if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                recognize_mode = False

            if not recognize_mode:
                if not mouse_control_enabled:
                    # 여기에서 마우스의 초기 위치를 설정
                    plocX, plocY = pyautogui.position()
                    mouse_control_enabled = True

                if len(lmList) != 0:
                    x1, y1 = lmList[8][1:]
                    x2, y2 = lmList[12][1:]
                    fingers = detector.fingersUp()

                    # 마우스 커서 이동 코드
                    if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening
                        pyautogui.moveTo(wScr - clocX, clocY)
                        # cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                    # 볼륨 조절
                    if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                        x_thumb, y_thumb = lmList[4][1], lmList[4][2]  # 엄지손가락 좌표
                        x_index, y_index = lmList[8][1], lmList[8][2]  # 검지손가락 좌표
                        distance = math.hypot(x_index - x_thumb, y_index - y_thumb)  # 두 손가락 사이의 유클리디안 거리 계산

                        # 거리에 따라 볼륨 조절 (예를 들어, 거리가 30px에서 200px 사이라고 가정)
                        vol = np.interp(distance, [30, 200], [0, 100])
                        set_volume(vol)  # 볼륨 설정 함수 호출

                        # 볼륨 상태를 이미지에 표시
                        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
                        volBar = np.interp(vol, [0, 100], [400, 150])
                        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                        cv2.putText(img, f'{int(vol)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)

                    # 클릭 이벤트 처리 코드
                    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                        length, img, lineInfo = detector.findDistance(8, 12, img)
                        if length < 40:
                            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                            clocX = plocX + (x3 - plocX) / smoothening
                            clocY = plocY + (y3 - plocY) / smoothening
                            pyautogui.mouseDown(wScr - clocX, clocY)
                            # cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                            plocX, plocY = clocX, clocY
                        else:
                            pyautogui.click()
                            # cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            pyautogui.sleep(0.5)

                else:
                    if mouse_control_enabled:
                        mouse_control_enabled = False  # 마우스 컨트롤 비활성화
                        recognize_mode = True
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
