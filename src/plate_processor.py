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


def find_contours_in_plate(thresh_plate):

    """번호판에서 윤곽선 검출"""

    

    # 윤곽선 검출

    contours, hierarchy = cv2.findContours(

        thresh_plate,                    # 이진화된 번호판 이미지

        mode=cv2.RETR_EXTERNAL,         # 가장 바깥쪽 윤곽선만 검출

        method=cv2.CHAIN_APPROX_SIMPLE  # 윤곽선 단순화

    )

    

    # 결과 시각화용 이미지 생성 (컬러)

    height, width = thresh_plate.shape

    contour_image = cv2.cvtColor(thresh_plate, cv2.COLOR_GRAY2BGR)

    

    # 모든 윤곽선을 다른 색으로 그리기

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 

              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]

    

    for i, contour in enumerate(contours):

        color = colors[i % len(colors)]  # 색상 순환

        cv2.drawContours(contour_image, [contour], -1, color, 2)

        

        # 윤곽선 번호 표시

        M = cv2.moments(contour)

        if M["m00"] != 0:

            cx = int(M["m10"] / M["m00"])

            cy = int(M["m01"] / M["m00"])

            cv2.putText(contour_image, str(i+1), (cx-5, cy+5), 

                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    

    # 결과 시각화

    plt.figure(figsize=(15, 5))

    

    plt.subplot(1, 3, 1)

    plt.imshow(thresh_plate, cmap='gray')

    plt.title('Binary Plate')

    plt.axis('off')

    

    plt.subplot(1, 3, 2)

    plt.imshow(contour_image)

    plt.title(f'Contours Detected: {len(contours)}')

    plt.axis('off')

    

    # 윤곽선 정보 표시

    plt.subplot(1, 3, 3)

    contour_info = np.zeros((height, width, 3), dtype=np.uint8)

    

    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        

        # 경계 사각형 그리기

        cv2.rectangle(contour_info, (x, y), (x+w, y+h), colors[i % len(colors)], 1)

        

        # 면적 정보 표시 (작은 글씨로)

        cv2.putText(contour_info, f'A:{int(area)}', (x, y-2), 

                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

    

    plt.imshow(contour_info)

    plt.title('Bounding Rectangles')

    plt.axis('off')

    

    plt.tight_layout()

    plt.show()

    

    # 윤곽선 정보 출력

    print("=== 윤곽선 검출 결과 ===")

    print(f"총 윤곽선 개수: {len(contours)}")

    

    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / h if h > 0 else 0

        print(f"윤곽선 {i+1}: 면적={area:.0f}, 크기=({w}×{h}), 비율={aspect_ratio:.2f}")

    

    return contours, contour_image


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


def save_processed_results(plate_name, gray_plate, enhanced_plate, thresh_plate, contour_result):

    """처리된 번호판 이미지들을 체계적으로 저장"""

    

    # 저장 폴더 생성

    save_dir = '../processed_plates'

    if not os.path.exists(save_dir):

        os.makedirs(save_dir)

    

    # 각 단계별 결과 저장

    cv2.imwrite(f'{save_dir}/{plate_name}_1_gray.png', gray_plate)

    cv2.imwrite(f'{save_dir}/{plate_name}_2_enhanced.png', enhanced_plate)  

    cv2.imwrite(f'{save_dir}/{plate_name}_3_threshold.png', thresh_plate)

    cv2.imwrite(f'{save_dir}/{plate_name}_4_contours.png', contour_result)

    

    print(f"처리 결과 저장 완료: {save_dir}/{plate_name}_*.png")

def process_extracted_plate(plate_name):

    """추출된 번호판의 완전한 처리 파이프라인"""

    

    print(f"=== {plate_name} 처리 시작 ===")

    

    # 1단계: 이미지 로드

    plate_img = load_extracted_plate(plate_name)

    if plate_img is None:

        return None

    

    # 2단계: 그레이스케일 변환

    gray_plate = convert_to_grayscale(plate_img)

    

    # 3단계: 대비 최대화

    enhanced_plate = maximize_contrast(gray_plate)

    

    # 4단계: 적응형 임계처리

    thresh_plate, _ = adaptive_threshold_plate(enhanced_plate)

    

    # 5단계: 윤곽선 검출

    contours, contour_result = find_contours_in_plate(thresh_plate)

    

    # 6단계: 결과 저장

    save_processed_results(plate_name, gray_plate, enhanced_plate, thresh_plate, contour_result)

    

    # 7단계: 처리 결과 요약

    potential_chars = prepare_for_next_step(contours, thresh_plate)

    print(f"처리 완료 - 검출된 윤곽선: {len(contours)}개, 잠재적 글자: {potential_chars}개")

    

    return {

        'original': plate_img,

        'gray': gray_plate, 

        'enhanced': enhanced_plate,

        'threshold': thresh_plate,

        'contours': len(contours),

        'potential_chars': potential_chars,

        'contour_result': contour_result

    }

# 배치 처리

def batch_process_plates():

    """extracted_plates 폴더의 모든 번호판 처리"""

    

    plate_dir = '../extracted_plates'

    if not os.path.exists(plate_dir):

        print(f"폴더를 찾을 수 없습니다: {plate_dir}")

        return {}

        

    plate_files = [f for f in os.listdir(plate_dir) if f.endswith('.png')]

    

    if len(plate_files) == 0:

        print("처리할 번호판 이미지가 없습니다.")

        return {}

    

    results = {}

    for plate_file in plate_files:

        plate_name = plate_file.replace('.png', '')

        result = process_extracted_plate(plate_name)

        if result:

            results[plate_name] = result

    

    print(f"\n=== 전체 처리 완료: {len(results)}개 번호판 ===")

    return results

plate_img = load_extracted_plate('plate_01')  # plate_01.png 로드

if plate_img is not None:

    gray_plate = convert_to_grayscale(plate_img)

    enhanced_plate = maximize_contrast(gray_plate)

    thresh_adaptive, thresh_otsu = adaptive_threshold_plate(enhanced_plate)
    compare_contour_modes(thresh_adaptive)
    compare_contour_modes(thresh_otsu)
