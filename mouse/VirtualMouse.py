import cv2
import numpy as np
import HandTracking as htm
import time
import pyautogui

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

# 변수 초기화, 마우스 컨트롤 비활성화 상태로 시작
mouse_control_enabled = False

while cap.isOpened():
    success, img = cap.read()
    if success:
        if not mouse_control_enabled:
            # 여기에서 마우스의 초기 위치를 설정
            plocX, plocY = pyautogui.position()
            mouse_control_enabled = True

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            fingers = detector.fingersUp()
            # cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            # 마우스 커서 이동 코드
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                pyautogui.moveTo(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # 클릭 이벤트 처리 코드
            if fingers[1] == 1 and fingers[2] == 1:
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

        # FPS 계산 및 표시
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# def control_mouse(img, detector, plocX, plocY, wCam, hCam, wScr, hScr, smoothening):
#     img = detector.findHands(img)
#     lmList, bbox = detector.findPosition(img)
#     clocX, clocY = plocX, plocY
#
#     if len(lmList) != 0:
#         x1, y1 = lmList[8][1:]
#         x2, y2 = lmList[12][1:]
#         fingers = detector.fingersUp()
#
#         if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
#             x3 = np.interp(x1, (100, wCam - 100), (0, wScr))
#             y3 = np.interp(y1, (100, hCam - 100), (0, hScr))
#             clocX = plocX + (x3 - plocX) / smoothening
#             clocY = plocY + (y3 - plocY) / smoothening
#             pyautogui.moveTo(wScr - clocX, clocY)
#             plocX, plocY = clocX, clocY
#
#         if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
#             length, img, lineInfo = detector.findDistance(8, 12, img)
#             if length < 40:
#                 pyautogui.mouseDown(wScr - clocX, clocY)
#             else:
#                 pyautogui.click()
#     return img, plocX, plocY
#
# def get_screen_size():
#     return pyautogui.size()