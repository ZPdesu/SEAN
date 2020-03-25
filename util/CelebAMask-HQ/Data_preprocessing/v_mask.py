import os
import numpy as np
from utils import make_folder
import skimage.io


def labelcolormap(N):
    if N == 19: # CelebAMask-HQ
        cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                     (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                     (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                     (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                     (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])



def colorize(gray_image, cmap):
    size = gray_image.shape
    color_image = np.zeros((size[0], size[1], 3), np.uint8)

    for label in range(0, len(cmap)):
        mask = (label == gray_image[:, :])
        color_image[:, :, 0][mask] = cmap[label][0]
        color_image[:, :, 1][mask] = cmap[label][1]
        color_image[:, :, 2][mask] = cmap[label][2]

    return color_image



folder_base = '/media/zhup/Data/CelebAMask-HQ/CelebAMaskHQ-mask'
folder_save = '/media/zhup/Data/CelebAMask-HQ/CelebAMaskHQ-mask-vis'
img_num = 30000

make_folder(folder_save)
my_cmp = labelcolormap(19)

for k in range(img_num):

    filename = os.path.join(folder_base, str(k) + '.png')
    if (os.path.exists(filename)):
        print(k + 1)
        im = skimage.io.imread(filename)
        im_vis = colorize(im,my_cmp)

    filename_save = os.path.join(folder_save, str(k) + '.png')
    print(filename_save)
    skimage.io.imsave(filename_save,im_vis)