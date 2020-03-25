
from glob import glob


import numpy as np
import os




layers_list = ['ACE.npy']


style_list = []


for cat_i in range(19):
    for layer_j in layers_list:
        tmp_list = glob('styles_test/style_codes/*/' + str(cat_i) + '/' + layer_j)
        style_list = []

        for k in tmp_list:
            style_list.append(np.load(k))

        if len(style_list) > 0:
            result = np.array(style_list).mean(0)
            save_folder = os.path.join('styles_test/mean_style_code/mean', str(cat_i))

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_name = os.path.join(save_folder, layer_j)
            np.save(save_name,result)

print(100)