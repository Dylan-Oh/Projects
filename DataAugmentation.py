import cv2
import os
import random
import numpy as np
import shutil

# 원본 데이터셋 경로
image_folder = "./wm_images"
# 전체 데이터셋 경로
merge_folder = "./full_wm_images"
# 증강된 데이터셋 경로
output_folder = "./augmented_images"
subfolders = ["test", "train", "val"]
classes = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "none", "Random", "Scratch"]

# test, train, val 폴더에 있던 이미지들을 모두 병합하는 함수
# 시험, 훈련, 검증 데이터는 데이터 증강 이후에 나눠져야 함
def merge_images(test_folder, train_folder, val_folder):
    # 전체 데이터셋 폴더 하위에 각 클래스 폴더 생성
    for cls in classes:
        os.makedirs(os.path.join(merge_folder, cls), exist_ok=True)
    
    # test, train, val 폴더 순회
    for subfolder in subfolders:
        # test, train, val 폴더 경로 저장
        src_folder = os.path.join(image_folder, subfolder)
        # 각 클래스 폴더 순회
        for cls in classes:
            # 각 클래스 폴더 경로 저장
            cls_src_folder = os.path.join(src_folder, cls)
            
            # 모든 파일 복사
            for file in os.listdir(cls_src_folder):
                copy_image(cls_src_folder, merge_folder, file, cls)
            print(subfolder + cls + ' 데이터 복사 완료')
                    
def argument_images():
    for cls in classes:
        # 증강 데이터셋 폴더 하위에 각 클래스 폴더 생성
        os.makedirs(os.path.join(output_folder, cls), exist_ok=True)
        # 클래스 폴더 경로
        cls_folder = os.path.join(merge_folder, cls)
        # 증강될 클래스 폴더 경로
        cls_output_folder = os.path.join(output_folder, cls)
        # 클래스별 이미지 리스트
        images = os.listdir(cls_folder)
        
        for image_name in images:
            # 원본 이미지 파일 경로
            original_image_path = os.path.join(cls_folder, image_name)
            # 원본 이미지 저장
            original_image = cv2.imread(original_image_path)

            copy_image(cls_folder, output_folder, image_name, cls)
            # 이미지 복사 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
            rotate_image(original_image, os.path.join(cls_output_folder, 'rotated_' + image_name))
            # 이미지 회전 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
            flip_image(original_image, os.path.join(cls_output_folder, 'mirror_' + image_name))
            # 이미지 반전 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
            width_left_shift_image(original_image, os.path.join(cls_output_folder, 'wleftshifted_' + image_name))
            # 이미지 좌측 이동 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
            width_right_shift_image(original_image, os.path.join(cls_output_folder, 'wrightshifted_' + image_name))
            # 이미지 우측 이동 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
            height_up_shift_image(original_image, os.path.join(cls_output_folder, 'hupshifted_' + image_name))
            # 이미지 위로 이동 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
            height_down_shift_image(original_image, os.path.join(cls_output_folder, 'hdownshifted_' + image_name))
            # 이미지 아래로 이동 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
            shear_image(original_image, os.path.join(cls_output_folder, 'shear_' + image_name))
            # 이미지 전단 이동 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
            channel_shift(original_image, os.path.join(cls_output_folder, 'cshifted_' + image_name))
            # 이미지 채널 이동 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
            zoomin_image(original_image, os.path.join(cls_output_folder, 'zoomin_' + image_name))
            # 이미지 확대 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
            zoomout_image(original_image, os.path.join(cls_output_folder, 'zoomout_' + image_name))
            # 이미지 축소 후 폴더에 이미지가 10,000개 이상이면 반복문 종료
            if len(os.listdir(cls_output_folder)) >= 10000:
                break
            
        # 증강 완료한 이미지 정보 저장
        first_augmentation_images = os.listdir(cls_output_folder)
        print(cls + ' : ', len(first_augmentation_images), '개 증강')
        # 증강된 데이터가 10,000개 보다 작을 경우 추가 증강 수행
        if len(first_augmentation_images) < 10000:
            print(cls + ' 10000개 미달... 추가 증강')
            for image_name in first_augmentation_images:
                # 증강된 폴더에 있는 원본 데이터
                original_image_path = os.path.join(cls_output_folder, image_name)
                original_image = cv2.imread(original_image_path)
                # 회전되지 않은 이미지를 회전하여 추가 증강 (wmd는 원본 데이터)
                if not image_name.startswith('rotated') and not image_name.startswith('wmd'):
                    # 회전 되지 않은 이미지 회전 후 폴더에 이미지가 10,000개 이상 있으면 종료
                    rotate_image(original_image, os.path.join(cls_output_folder, 'rotated_' + image_name))
                    if len(os.listdir(cls_output_folder)) >= 10000:
                        break
                    
                # 대칭되지 않은 이미지를 대칭하여 추가 증강 (wmd는 원본 데이터)
                if not image_name.startswith('mirror') and not image_name.startswith('wmd'):
                    # 대칭되지 않은 이미지 대칭 후 폴더에 이미지가 10,000개 이상 있으면 종료
                    flip_image(original_image, os.path.join(cls_output_folder, 'mirror_' + image_name))
                    if len(os.listdir(cls_output_folder)) >= 10000:
                        break
                    
                # 전단 이동되지 않은 이미지를 전단 이동하여 추가 증강 (wmd는 원본 데이터)
                if not image_name.startswith('shear') and not image_name.startswith('wmd'):
                    # 전단 이동되지 않은 이미지 전단 이동 후 폴더에 이미지가 10,000개 이상 있으면 종료
                    shear_image(original_image, os.path.join(cls_output_folder, 'shear_' + image_name))
                    if len(os.listdir(cls_output_folder)) >= 10000:
                        break
                    
                # 채널 이동되지 않은 이미지를 채널 이동하여 추가 증강 (wmd는 원본 데이터)
                if not image_name.startswith('cshifted') and not image_name.startswith('wmd'):
                    # 채널 이동되지 않은 이미지 채널 이동 후 폴더에 이미지가 10,000개 이상 있으면 종료
                    channel_shift(original_image, os.path.join(cls_output_folder, 'cshifted_' + image_name))
                    if len(os.listdir(cls_output_folder)) >= 10000:
                        break
                    
                # 확대/축소되지 않은 이미지를 확대/축소하여 추가 증강 (wmd는 원본 데이터)
                if not image_name.startswith('zoom') and not image_name.startswith('wmd'):
                    # 확대/축소되지 않은 이미지 확대/축소 후 폴더에 이미지가 10,000개 이상 있으면 종료
                    zoomin_image(original_image, os.path.join(cls_output_folder, 'zoomin_' + image_name))
                    if len(os.listdir(cls_output_folder)) >= 10000:
                        break
                    
                    zoomout_image(original_image, os.path.join(cls_output_folder, 'zoomout_' + image_name))
                    if len(os.listdir(cls_output_folder)) >= 10000:
                        break
                
                # 이동되지 않은 이미지를 이동하여 추가 증강 (wmd는 원본 데이터)
                if not image_name.startswith('wleft') and not image_name.startswith('wright') and not image_name.startswith('hup') and not image_name.startswith('hdown') and not image_name.startswith('wmd'):
                    # 각 이동 되지 않은 이미지 를 이동 후 폴더에 이미지가 10,000개 이상 있으면 종료
                    width_left_shift_image(original_image, os.path.join(cls_output_folder, 'wleftshifted_' + image_name))
                    if len(os.listdir(cls_output_folder)) >= 10000:
                        break
                    
                    width_right_shift_image(original_image, os.path.join(cls_output_folder, 'wrightshifted_' + image_name))
                    if len(os.listdir(cls_output_folder)) >= 10000:
                        break
                    
                    height_up_shift_image(original_image, os.path.join(cls_output_folder, 'hupshifted_' + image_name))
                    if len(os.listdir(cls_output_folder)) >= 10000:
                        break
                    
                    height_down_shift_image(original_image, os.path.join(cls_output_folder, 'hdownshifted_' + image_name))
                    if len(os.listdir(cls_output_folder)) >= 10000:
                        break
            print(cls + ' : ', len(os.listdir(cls_output_folder)), '개 증강')
                    
        
       

# 이미지 복사 함수
def copy_image(original_folder, result_folder, image_path, class_name):
    # 원본 이미지 경로 저장
    original_image = os.path.join(original_folder, image_path)
    # 복사 결과 이미지 경로 저장
    result_image = os.path.join(result_folder, class_name, image_path)
    # 이미지 복사
    shutil.copy(original_image, result_image)
        
# +-10도 무작위 회전
def rotate_image(original_image, result_image):

    # 이미지의 높이와 너비 가져오기
    height, width = original_image.shape[:2]

    # 무작위 회전 각도 생성 (-10도에서 +10도)
    angle = random.randint(-10, 10)

    # 회전 중심점 계산
    center_x = width // 2
    center_y = height // 2

    # 회전 변환 행렬 계산
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

    # 이미지 회전
    rotated_image = cv2.warpAffine(original_image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

    # 회전된 이미지를 결과 경로에 저장
    cv2.imwrite(result_image, rotated_image)
    
# 좌우대칭
def flip_image(original_image, result_image):
    mirrored_image = cv2.flip(original_image, 1)
    
    cv2.imwrite(result_image, mirrored_image)

# 넓이 왼쪽으로 20% 이동
def width_left_shift_image(original_image, result_image):
    height, width = original_image.shape[:2]
    width_shift_amount = int(width * 0.2)
    
    # 변환 행렬 생성
    M = np.float32([[1, 0, -width_shift_amount], [0, 1, 0]])
    
    # 이미지 이동
    width_shifted_image = cv2.warpAffine(original_image, M, (width, height))
    
    cv2.imwrite(result_image, width_shifted_image)
    
# 넓이 오른쪽으로 20% 이동
def width_right_shift_image(original_image, result_image):
    height, width = original_image.shape[:2]
    width_shift_amount = int(width * 0.2)
    
    # 변환 행렬 생성
    M = np.float32([[1, 0, width_shift_amount], [0, 1, 0]])
    
    # 이미지 이동
    width_shifted_image = cv2.warpAffine(original_image, M, (width, height))
    
    cv2.imwrite(result_image, width_shifted_image)

# 위로 15% 이동
def height_up_shift_image(original_image, result_image):
    height, width = original_image.shape[:2]
    height_shift_amount = int(height * 0.15)
        
    # 이미지를 위로 이동시킵니다.
    shifted_image = cv2.warpAffine(original_image, 
                                   np.float32([[1, 0, 0], [0, 1, -height_shift_amount]]), 
                                   (width, height))
    
    cv2.imwrite(result_image, shifted_image)
    
# 아래로 15% 이동
def height_down_shift_image(original_image, result_image):
    height, width = original_image.shape[:2]
    height_shift_amount = int(height * 0.15)
        
    # 이미지를 아래로 이동시킵니다.
    shifted_image = cv2.warpAffine(original_image, 
                                   np.float32([[1, 0, 0], [0, 1, height_shift_amount]]), 
                                   (width, height))
    
    cv2.imwrite(result_image, shifted_image)
    

# 전단 10% 변환
def shear_image(original_image, result_image):
    # 이미지의 높이와 너비 가져오기
    height, width = original_image.shape[:2]

    # 전단 변환 행렬 생성
    shear_matrix = np.float32([[1, 0.1, 0], [0, 1, 0]])

    # 이미지에 전단 변환 적용
    sheared_image = cv2.warpAffine(original_image, shear_matrix, (width + int(height*0.1), height))

    cv2.imwrite(result_image, sheared_image)
    
# 채널 10% 이동
def channel_shift(original_image, result_image):
    # 채널 분할
    b, g, r = cv2.split(original_image)

    # 이동할 양 계산
    chanel_shift_amount = int(original_image.shape[1] * 0.1)

    # 각 채널 이동
    b_shifted = cv2.copyMakeBorder(b, 0, 0, chanel_shift_amount, 0, cv2.BORDER_WRAP)
    g_shifted = cv2.copyMakeBorder(g, 0, 0, chanel_shift_amount, 0, cv2.BORDER_WRAP)
    r_shifted = cv2.copyMakeBorder(r, 0, 0, chanel_shift_amount, 0, cv2.BORDER_WRAP)

    # 이동된 채널 병합
    chanel_shifted_image = cv2.merge((b_shifted, g_shifted, r_shifted))

    cv2.imwrite(result_image, chanel_shifted_image)
    
# 이미지 10% 확대
def zoomin_image(original_image, result_image):
    # 이미지의 높이와 너비 가져오기
    height, width = original_image.shape[:2]

    # 확대된 이미지 크기 계산
    new_width = int(width * (1.1))
    new_height = int(height * (1.1))

    # 이미지 확대
    zoom_image = cv2.resize(original_image, (new_width, new_height))

    cv2.imwrite(result_image, zoom_image)
    
# 이미지 10% 축소
def zoomout_image(original_image, result_image):
    # 이미지의 높이와 너비 가져오기
    height, width = original_image.shape[:2]

    # 확대된 이미지 크기 계산
    new_width = int(width * (0.9))
    new_height = int(height * (0.9))

    # 이미지 확대
    zoom_image = cv2.resize(original_image, (new_width, new_height))

    cv2.imwrite(result_image, zoom_image)


# 전체 데이터셋 폴더가 존재하지 않는다면 폴더 생성 후 이미지 병합
if not os.path.exists(merge_folder):
    os.makedirs(merge_folder)
    merge_images(os.path.join(image_folder, subfolders[0]), 
             os.path.join(image_folder, subfolders[1]), 
             os.path.join(image_folder, subfolders[2]))

# 증강 데이터셋 폴더가 존재하지 않는다면 폴더 생성 후 이미지 증강
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    argument_images()
    

