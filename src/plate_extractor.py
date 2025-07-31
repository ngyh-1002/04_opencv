import cv2
import numpy as np
import datetime
import os


win_name = "License Plate Extractor"
img = cv2.imread("../img/car_02.jpg")
rows, cols = img.shape[:2]
draw = img.copy()
pts_cnt = 0
pts = np.zeros((4,2), dtype=np.float32)

def onMouse(event, x, y, flags, param):  #마우스 이벤트 콜백 함수 구현 ---① 
    global  pts_cnt                     # 마우스로 찍은 좌표의 갯수 저장
    if event == cv2.EVENT_LBUTTONDOWN:  
        cv2.circle(draw, (x,y), 10, (0,255,0), -1) # 좌표에 초록색 동그라미 표시
        cv2.imshow(win_name, draw)

        pts[pts_cnt] = [x,y]            # 마우스 좌표 저장
        pts_cnt+=1
        if pts_cnt == 4:                       # 좌표가 4개 수집됨 
            # 좌표 4개 중 상하좌우 찾기 ---② 
            sm = pts.sum(axis=1)                 # 4쌍의 좌표 각각 x+y 계산
            diff = np.diff(pts, axis = 1)       # 4쌍의 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]         # x+y가 가장 값이 좌상단 좌표
            bottomRight = pts[np.argmax(sm)]     # x+y가 가장 큰 값이 우하단 좌표
            topRight = pts[np.argmin(diff)]     # x-y가 가장 작은 것이 우상단 좌표
            bottomLeft = pts[np.argmax(diff)]   # x-y가 가장 큰 값이 좌하단 좌표

            # 변환 전 4개 좌표 
            pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])
            
            # 변환 이미지 프레임 크기 초기화
            width = 300
            height = 150
            # 변환 후 4개 좌표
            pts2 = np.float32([[0,0], [width-1,0], 
                                [width-1,height-1], [0,height-1]])
            
            # 변환 행렬 계산 
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (width, height))
            
            # 현재 타임스탬프 생성
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # 'extracted_plates' 디렉토리 내에서 현재 타임스탬프와 일치하는 파일들을 찾아서 순번 결정
            #    (아직 디렉토리가 없다면 생성)
            output_dir = "../extracted_plates"
            os.makedirs(output_dir, exist_ok=True) # 디렉토리가 없으면 생성
            
            # 현재 타임스탬프를 포함하는 파일 중 가장 높은 순번을 찾습니다.
            latest_sequence = 0
            for filename in os.listdir(output_dir):
                if filename.startswith(f"plate_{timestamp}_") and filename.endswith(".png"):
                    try:
                        # 파일명에서 순번 부분만 추출 (예: 001, 002)
                        seq_str = filename.split('_')[-1].split('.')[0]
                        current_seq = int(seq_str)
                        if current_seq > latest_sequence:
                            latest_sequence = current_seq
                    except ValueError:
                        # 순번 부분이 숫자가 아닌 경우 무시
                        continue

            # 다음 순번 결정 (만약 해당 타임스탬프의 첫 파일이라면 1, 아니면 기존 순번 + 1)
            next_sequence = latest_sequence + 1

            # 3. 최종 파일명 생성 (예: extracted_plates/plate_20250731_123000_001.jpg)
            #    순번은 항상 세 자리로 포맷팅합니다 (예: 1 -> 001, 10 -> 010)
            final_filename = f"{output_dir}/plate_{timestamp}_{next_sequence:03d}.png"
            
            # 파일 저장

            success = cv2.imwrite(final_filename, result)
            if success:
                print(f"번호판 저장 완료: {final_filename}")

                cv2.imshow('Extracted Plate', result)

            else:

                print("저장 실패!")


            
cv2.imshow(win_name, img)
cv2.setMouseCallback(win_name, onMouse)    # 마우스 콜백 함수를 GUI 윈도우에 등록 ---④
cv2.waitKey(0)
cv2.destroyAllWindows()         