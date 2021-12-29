import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

comp_a = Image.open('comp_a.png')
comp_b = Image.open('comp_b.png')

grv_loaded = pd.read_csv('grv_processed.csv')
plt.hist(np.log(grv_loaded.area_comp_a_km2))


type(comp_b)
color_codes = [[250, 250, 150],
                [250, 250, 120],
                [250, 250, 90],
                [250, 220, 90],
                [250, 190, 90],
                [250, 160, 90],
                [250, 130, 90],
                [200, 130, 60],
                [180, 130, 60],
                [160, 130, 90],
                [160, 130, 60],
                [160, 130, 30]]

len(color_codes)
comp_a_arr = np.asarray(comp_a)
comp_a_arr.shape
color_codes = pd.Series(color_codes)
owc_cc = [20, 100, 200]

layer_bottoms = np.arange(9800, 11000, 100)

grv = pd.DataFrame(columns=['layer_bottom', 'color_code', 'area_comp_a_pxl', 'area_comp_b_pxl'])
grv['layer_bottom'] = layer_bottoms
# grv['color_code'] = grv['color_code'].astype('object')
# grv.dtypes
grv['color_code'] = color_codes
grv['area_comp_a_pxl'], grv['area_comp_b_pxl'] = np.zeros(len(grv)), np.zeros(len(grv))


grv
grv.to_csv('grv.csv', index=False)

# for pixel in comp_a.getdata():
#     for i in range(len(grv)):
#         if pixel[:-1] == tuple(grv.at[i, 'color_code']):
#             grv.at[i, 'area_comp_a_pxl'] += 1

for i in range(len(grv)):

    for pixel in comp_a.getdata():
        if pixel[:-1] == tuple(grv.at[i, 'color_code']):
            grv.at[i, 'area_comp_a_pxl'] += 1

    for pixel in comp_b.getdata():
        if pixel[:-1] == tuple(grv.at[i, 'color_code']):
            grv.at[i, 'area_comp_b_pxl'] += 1

grv['area_comp_a_pxl']
grv['area_comp_a_km2'] = grv['area_comp_a_pxl'] / 305**2
grv['area_comp_b_km2'] = grv['area_comp_b_pxl'] / 305**2
grv['area_comp_a_ft2'] = grv['area_comp_a_km2'] * 1.076e+7
grv['area_comp_b_ft2'] = grv['area_comp_b_km2'] * 1.076e+7

color_codes = np.array(color_codes).reshape(len(color_codes), 1)
color_codes.shape
flat_color_codes = color_codes.flatten()
flat_color_codes
img = np.array([[row for row in color_codes] for i in range(2)])
img.shape

%matplotlib qt
plt.imshow(img)
plt.yticks([], [])
plt.xticks(np.arange(12), layer_bottoms, rotation=45, fontsize=12)

#
# grv
# grv.to_csv('grv.csv')
#
# grv = pd.read_csv('grv.csv')
# grv_ld = pd.read_csv('grv.csv').iloc[:, 1:]

for i in range(1, len(grv)):
    grv.iloc[i, 2:] += grv.iloc[i-1, 2:]

grv.to_csv('grv_processed.csv', index=False)
grv
