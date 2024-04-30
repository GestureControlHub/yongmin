import cv2
import numpy as np
import pyautogui
import time
import os

from learn import train_model as tm
from mouse import HandTracking as htm
# from mouse import VirtualMouse as vm
import mediapipe as mp

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

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)

wScr, hScr = pyautogui.size()

mouse_control_enabled = False

recognize_mode = True
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

                    if gesture_name == "paper":
                        if os.name == 'posix':
                            pyautogui.hotkey('command', 'c')
                        else:
                            pyautogui.hotkey('ctrl', 'c')
                        cv2.putText(img, 'copy!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
                        time.sleep(1)
                    elif gesture_name == "rock":
                        if os.name == 'posix':
                            pyautogui.hotkey('command', 'v')
                        else:
                            pyautogui.hotkey('ctrl', 'v')
                        cv2.putText(img, 'paste!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
                        time.sleep(1)
                    elif gesture_name == "scissors":
                        print(gesture_name)
                        time.sleep(1)
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
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                    # 클릭 이벤트 처리 코드
                    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                        length, img, lineInfo = detector.findDistance(8, 12, img)
                        if length < 40:
                            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                            clocX = plocX + (x3 - plocX) / smoothening
                            clocY = plocY + (y3 - plocY) / smoothening
                            pyautogui.mouseDown(wScr - clocX, clocY)
                            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                            plocX, plocY = clocX, clocY
                        else:
                            pyautogui.click()
                            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                            pyautogui.sleep(0.5)

                else:
                    if mouse_control_enabled:
                        mouse_control_enabled = False  # 마우스 컨트롤 비활성화
                        recognize_mode = True

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
