import cv2
import numpy as np
import sys
import os
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

"""
directory
-[여기를 dir_path로] 
    -dent       -images
                -masks
    -scratch    -images
                -masks
    -spacing    -images
                -masks

- Label 추가하는 코드랑 annotation file 추가하는 코드도 추가할 필요 OO

read_data
    - dir_path 는 해당 프로젝트 디렉토리에서 data 접근 전 위치로 하면 됨
"""
def read_data(dir_path = os.getcwd()+'/data/segmentation/'):
    print(os.getcwd())
    print(os.path.dirname(os.path.realpath(__file__)))

 # 확인해보고 싶은 데이터 CLASS
    cls = 'dent'
    cls2 = 'scratch'
    # cls = 'scratch'
    # cls = 'spacing'
    img_path = dir_path+cls+'/valid/images/'
    mask_path = dir_path + cls + '/valid/masks/'
    mask2_path = dir_path + cls2 + '/valid/masks/'

 # 해당 directory file 긁어옴
    image_list = os.listdir(img_path)
    mask_list = os.listdir(mask_path)
    mask2_list = os.listdir(mask2_path)
    print(os.listdir(img_path))
    print(len(os.listdir(img_path)))

# ==================================================================================
# 이부분에서 mask, img 합쳐주면 될거같아!
    for idx, (image, mask, mask2) in enumerate(zip(image_list, mask_list, mask2_list)):
      # 일단 한장

        # if idx<5:
        print(f'{idx+1} 번째 : {image}, {mask}')

        img = cv2.imread(img_path+image)  # 오른쪽 사진
        msk = cv2.imread(mask_path+mask)  # 왼쪽 사진
        msk2 = cv2.imread(mask2_path+mask2)

        print(f'image shape : {img.shape}, mask shape : {msk.shape}')
        # img_show = cv2.hconcat([img,msk])
        # print(img_show.shape)

        # mask 부분
        _msk = msk[:,:,0]
        _msk = cv2.cvtColor(_msk, cv2.COLOR_BGR2RGB)
        # _msk = cv2.bitwise_not(_msk)
        # _msk_ = np.zeros(_msk.shape)

        # mask2
        _msk2 = msk2[:, :, 0]
        _msk2 = cv2.cvtColor(_msk2, cv2.COLOR_BGR2RGB)
        # _msk2 = cv2.bitwise_not(_msk2)
        # _msk2_ = np.zeros(_msk2.shape)

        print('shape : ',_msk2.shape)

        color_mask = cv2.applyColorMap(_msk, cv2.COLORMAP_OCEAN)
        color_mask2 = cv2.applyColorMap(_msk2, cv2.COLORMAP_OCEAN)
        print('color_mask2 : ',color_mask2.shape)
        # color_mask = _msk
        # color_mask2 = _msk2


        # mask 색깔 단일로 만들어야 보기 좋음
        color_mask[:, :, 0] = 0 #np.zeros(_msk.shape)
        color_mask[:, :, 1] = 0 #np.zeros(_msk.shape)
        # color_mask[:, :, 2] = np.zeros(_msk.shape)
        # color_mask2[:, :, 0] = np.ones(_msk2.shape)
        color_mask2[:, :, 1] = 0 #np.zeros(_msk2.shape)
        color_mask2[:, :, 2] = 0#np.zeros(_msk2.shape)


        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_show = img
        img_show = cv2.addWeighted(img, 1, color_mask, 0.2, 0.0)
        img_show = cv2.addWeighted(img_show, 1, color_mask2, 0.4, 0.0)

      # save file part
      #   save_path = './data/answer1/'
      #   save_file = save_path+image
      #   print(save_file)
      #   cv2.imwrite(save_file, img_show)


        #
        _show = 0
        if(_show):
            plt.imshow(img_show)
            plt.waitforbuttonpress(-1)
            plt.close()

def read_spacing(dir_path = os.getcwd()+'/data/segmentation/'):
    print(os.getcwd())
    print(os.path.dirname(os.path.realpath(__file__)))

 # 확인해보고 싶은 데이터 CLASS
 #    cls = 'dent'
 #    cls2 = 'scratch'
    # cls = 'scratch'
    cls = 'spacing'
    img_path = dir_path+cls+'/valid/images/'
    mask_path = dir_path + cls + '/valid/masks/'
    # mask2_path = dir_path + cls2 + '/valid/masks/'

 # 해당 directory file 긁어옴
    image_list = os.listdir(img_path)
    mask_list = os.listdir(mask_path)
    # mask2_list = os.listdir(mask2_path)
    print(os.listdir(img_path))
    print(len(os.listdir(img_path)))

    for idx, (image, mask) in enumerate(zip(image_list, mask_list)):
      # 일단 한장

        # if idx<5:
        print(f'{idx+1} 번째 : {image}, {mask}')
        try:
            img = cv2.imread(img_path+image)  # 오른쪽 사진
            msk = cv2.imread(mask_path+mask)  # 왼쪽 사진

            # print(f'image shape : {img.shape}, mask shape : {msk.shape}')
            # img_show = cv2.hconcat([img,msk])
            # print(img_show.shape)

            # mask 부분
            _msk = msk[:,:,0]
            _msk = cv2.cvtColor(_msk, cv2.COLOR_BGR2RGB)
            # _msk = cv2.bitwise_not(_msk)
            # _msk_ = np.zeros(_msk.shape)

            color_mask = cv2.applyColorMap(_msk, cv2.COLORMAP_OCEAN)

            # mask 색깔 단일로 만들어야 보기 좋음
            color_mask[:, :, 0] = 0 #np.zeros(_msk.shape)
            # color_mask[:, :, 1] = 0 #np.zeros(_msk.shape)
            color_mask[:, :, 2] = 0#np.zeros(_msk.shape)



            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img_show = img

            img_show = cv2.addWeighted(img, 1, color_mask, 0.2, 0.0)
            # img_show = cv2.addWeighted(img_show, 1, color_mask2, 0.4, 0.0)

          # save file part
            # 저장할 때는 BRG2RGB 필요 없음 show할 때만 필요
            save_path = './data/answer_spacing/'
            save_file = save_path+image
            print(save_file)
            cv2.imwrite(save_file, img_show)


            #
            _show = 0
            if(_show):
                plt.imshow(img_show)
                plt.waitforbuttonpress(-1)
                plt.close()
      # ./DS_store?? 이런게 잡혀서...
        except:
            continue


