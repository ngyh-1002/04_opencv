import cv2

import numpy as np

from matplotlib import pyplot as plt

import os

def load_extracted_plate(plate_name):

    """추출된 번호판 이미지 로드"""

    plate_path = f'../extracted_plates/{plate_name}.png'

    if os.path.exists(plate_path):

        plate_img = cv2.imread(plate_path)

        print(f"번호판 이미지 로드 완료: {plate_img.shape}")

        return plate_img

    else:

        print(f"파일을 찾을 수 없습니다: {plate_path}")

        return None

def convert_to_grayscale(plate_img):

    """번호판을 그레이스케일로 변환"""

    

    # BGR을 그레이스케일로 변환

    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    

    # 결과 비교 시각화

    plt.figure(figsize=(12, 4))

    

    plt.subplot(1, 2, 1)

    plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))

    plt.title('Original Extracted Plate')

    plt.axis('off')

    

    plt.subplot(1, 2, 2)

    plt.imshow(gray_plate, cmap='gray')

    plt.title('Grayscale Plate')

    plt.axis('off')

    

    plt.tight_layout()

    plt.show()

    

    return gray_plate
def maximize_contrast(gray_plate):

    """번호판의 글자 대비 최대화"""

    

    # 모폴로지 연산용 구조화 요소 (번호판용으로 작게 설정)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 3x3 → 2x2로 축소

    

    # Top Hat: 밝은 세부사항 (흰 배경) 강조

    tophat = cv2.morphologyEx(gray_plate, cv2.MORPH_TOPHAT, kernel)

    

    # Black Hat: 어두운 세부사항 (검은 글자) 강조  

    blackhat = cv2.morphologyEx(gray_plate, cv2.MORPH_BLACKHAT, kernel)

    

    # 대비 향상 적용

    enhanced = cv2.add(gray_plate, tophat)

    enhanced = cv2.subtract(enhanced, blackhat)

    

    # 추가: 히스토그램 균등화로 대비 더욱 향상

    enhanced = cv2.equalizeHist(enhanced)

    

    # 결과 비교

    plt.figure(figsize=(15, 4))

    

    plt.subplot(1, 4, 1)

    plt.imshow(gray_plate, cmap='gray')

    plt.title('Original Gray')

    plt.axis('off')

    

    plt.subplot(1, 4, 2)

    plt.imshow(tophat, cmap='gray')

    plt.title('Top Hat')

    plt.axis('off')

    

    plt.subplot(1, 4, 3)

    plt.imshow(blackhat, cmap='gray')

    plt.title('Black Hat')

    plt.axis('off')

    

    plt.subplot(1, 4, 4)

    plt.imshow(enhanced, cmap='gray')

    plt.title('Enhanced Contrast')

    plt.axis('off')

    

    plt.tight_layout()

    plt.show()

    

    return enhanced


plate_img = load_extracted_plate('plate_01')  # plate_01.png 로드

if plate_img is not None:

    gray_plate = convert_to_grayscale(plate_img)

    enhanced_plate = maximize_contrast(gray_plate)
