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

def adaptive_threshold_plate(enhanced_plate):

    """번호판 전용 적응형 임계처리"""

    

    # 1단계: 가벼운 블러링 (노이즈 제거, 글자는 보존)

    blurred = cv2.GaussianBlur(enhanced_plate, (3, 3), 0)  # 5x5 → 3x3로 축소

    

    # 2단계: 번호판 최적화 적응형 임계처리

    thresh_adaptive = cv2.adaptiveThreshold(

        blurred,

        maxValue=255,

        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        thresholdType=cv2.THRESH_BINARY,  # BINARY_INV 대신 BINARY 사용

        blockSize=11,  # 19 → 11로 축소 (번호판 크기에 맞춤)

        C=2           # 9 → 2로 축소 (세밀한 조정)

    )

    

    # 3단계: Otsu 임계처리와 비교

    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    

    # 4단계: 결과 비교

    plt.figure(figsize=(16, 4))

    

    plt.subplot(1, 4, 1)

    plt.imshow(enhanced_plate, cmap='gray')

    plt.title('Enhanced Plate')

    plt.axis('off')

    

    plt.subplot(1, 4, 2)

    plt.imshow(blurred, cmap='gray')

    plt.title('Blurred')

    plt.axis('off')

    

    plt.subplot(1, 4, 3)

    plt.imshow(thresh_adaptive, cmap='gray')

    plt.title('Adaptive Threshold')

    plt.axis('off')

    

    plt.subplot(1, 4, 4)

    plt.imshow(thresh_otsu, cmap='gray')

    plt.title('Otsu Threshold')

    plt.axis('off')

    

    plt.tight_layout()

    plt.show()

    

    return thresh_adaptive, thresh_otsu


def compare_contour_modes(thresh_plate):

    """다양한 윤곽선 검출 모드 비교"""

    

    # 여러 모드로 윤곽선 검출

    modes = [

        (cv2.RETR_EXTERNAL, "EXTERNAL"),

        (cv2.RETR_LIST, "LIST"), 

        (cv2.RETR_TREE, "TREE")

    ]

    

    plt.figure(figsize=(15, 5))

    

    for i, (mode, mode_name) in enumerate(modes):

        contours, _ = cv2.findContours(thresh_plate, mode, cv2.CHAIN_APPROX_SIMPLE)

        

        result_img = cv2.cvtColor(thresh_plate, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 1)

        

        plt.subplot(1, 3, i+1)

        plt.imshow(result_img)

        plt.title(f'{mode_name}: {len(contours)} contours')

        plt.axis('off')
        prepare_for_next_step(contours, thresh_plate)
    

    plt.tight_layout()

    plt.show()


def prepare_for_next_step(contours, thresh_plate):

    """다음 단계(글자 분석)를 위한 기본 정보 준비"""

    

    print("=== 다음 단계 준비 ===")

    

    # 윤곽선이 충분히 검출되었는지 확인

    if len(contours) < 5:

        print("윤곽선이 적게 검출되었습니다. 전처리 단계를 재검토하세요.")

    elif len(contours) > 20:

        print("윤곽선이 너무 많이 검출되었습니다. 노이즈 제거가 필요할 수 있습니다.")

    else:

        print("적절한 수의 윤곽선이 검출되었습니다.")

    

    # 잠재적 글자 후보 개수 추정

    potential_chars = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if 30 < area < 2000:  # 글자 크기 범위 추정

            potential_chars += 1

    

    print(f"잠재적 글자 후보: {potential_chars}개")

    

    return potential_chars

plate_img = load_extracted_plate('plate_01')  # plate_01.png 로드

if plate_img is not None:

    gray_plate = convert_to_grayscale(plate_img)

    enhanced_plate = maximize_contrast(gray_plate)

    thresh_adaptive, thresh_otsu = adaptive_threshold_plate(enhanced_plate)
    compare_contour_modes(thresh_adaptive)
    compare_contour_modes(thresh_otsu)
