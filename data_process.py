"""
Already made
- 기존에 만들어 놓은 COCO Format으로 바꾸는 코드 추가 필요
- mask, data 겹쳐서 확인

New
- [Done on 22 Nov]mask 겹치는 코드 필요
- 분포 확인
- image crop(resize) to 512x512 size

mask 기준
- dent 를 scratch 위에
- scratch 를 dent 위에
- dent & scratch 라는 label 새로 만들기

"""
##
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


##
def image_resize():
    pass


def image_crop():
    pass


def combine_mask(msk_dir1, msk_dir2):
    """
    image histogram : http://www.gisdeveloper.co.kr/?p=6634

    dent 의 damage 가 더 critical 한 것이기 때문에 dent mask가 scatch mask를 덮어씌게 할 예정

    overlaped part can be get by & operator
    https://python.plainenglish.io/how-to-find-an-intersection-between-two-matrices-easily-using-numpy-30263373b546
    """
    # mask directory : msk1_dir, msk2_dir
    msk_list1 = os.listdir(msk_dir1)
    msk_list2 = os.listdir(msk_dir2)
    msk_list1.sort()
    msk_list2.sort()

    if(len(msk_list1)!=len(msk_list2)):
        print('The number of masks are different something wrong')
        return 0

# Making directories for new mask
    result_dir = '../data/new_mask'
    if 'train' in msk_dir1:
        tr_te_va = 'train'
    elif 'test' in msk_dir1:
        tr_te_va = 'test'
    elif 'valid' in msk_dir1:
        tr_te_va = 'valid'
    result_dir = os.path.join(result_dir, tr_te_va)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    # ii = 0
    for idx, (msk1_name, msk2_name) in enumerate(zip(msk_list1, msk_list2)):
        msk1 = cv2.imread(os.path.join(msk_dir1, msk1_name), 0)  # read image as gray scale
        msk2 = cv2.imread(os.path.join(msk_dir2, msk2_name), 0)

     # mask : make value clear
        msk1 = np.array(msk1)
        msk2 = np.array(msk2)
        msk1 = np.where(msk1 < 10, 0, msk1)  # 10이하의 값들은 0으로 바꿔줌
        msk1 = np.where(msk1 > 245, 255, msk1)  # 245이상의 값들은 255으로 바꿔줌
        msk2 = np.where(msk2 < 10, 0, msk2)  # 10이하의 값들은 0으로 바꿔줌
        msk2 = np.where(msk2 > 245, 255, msk2)  # 245이상의 값들은 255으로 바꿔줌

     # dent, scratch 어떤걸 위에 올릴지 정하는 부분
        msk_new = np.zeros((msk2.shape[0], msk2.shape[1], 3))  # new mask shape : mask size with 3 channel
        msk3 = msk1 * msk2 # overlapped area : msk3
        msk3_p = np.where(msk3 != 0) # overlap area to 0
        # msk1[msk3_p] = 0 # overlap area process 1
        msk2[msk3_p] = 0 # overlap area process 2
        msk_new[:, :, 0] = msk2 # msk2 color : (255, 0, 0)
        msk_new[:, :, 2] = msk1 # msk1 color : (0, 0, 255)

        savename = os.path.join(result_dir, msk1_name.split(".")[0]) + '.png'
        cv2.imwrite(savename, msk_new)

        # pixel count
        n1 = len(np.where(msk1 != 0)[0])
        n2 = len(np.where(msk2 != 0)[0])

    # show mask by plt
        # if ii<10: #
        #     if(n1==0 and n2==0):
        #         pass
        #     else:# (n1 != n2):
        #         print(f'{msk1_name}')
        #         print(f'n1 {n1}  n2 {n2}')
        #         ii += 1
        #
        #         plt.figure(figsize=(10,10))
        #         plt.subplot(231)
        #         plt.imshow((msk1 * 255).astype(np.uint8))
        #         # plt.imshow(msk1)
        #
        #         plt.subplot(232)
        #         img_ = cv2.imread(os.path.join(data_dir,'dent','train','images',msk1_name))
        #         img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        #         plt.imshow(img_)
        #
        #         plt.subplot(233)
        #         plt.imshow((msk2 * 255).astype(np.uint8))
        #
        #         plt.subplot(212)
        #         # plt.imshow(msk3)
        #         plt.imshow(msk_new)
        #
        #         plt.show()






## 실행하는 부분

# for c in ['dent', 'scratch', 'spacing']:
#     for t in ['test', 'valid', 'train']:
#         file_list = os.listdir(os.path.join(data_dir,c,t,'masks'))
#         print(len(file_list))
data_dir = "../data/accida_segmentation_dataset_v1"

msk_dir1 = os.path.join(data_dir,'dent','test','masks')
msk_dir2 = os.path.join(data_dir,'scratch','test','masks')

combine_mask(msk_dir1, msk_dir2)








