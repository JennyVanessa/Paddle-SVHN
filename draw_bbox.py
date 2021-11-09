import os
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_attrs(digit_struct_mat_file, index):
    """
    Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
    """
    attrs = {}
    f = digit_struct_mat_file
    item = f['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = f[item][key]
        values = [f[attr[i].item()][0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
        attrs[key] = values
    return attrs



path_to_dir = 'data/test'
path_to_digit_struct_mat_file = os.path.join(path_to_dir, 'digitStruct.mat')

path_to_image_file = os.path.join(path_to_dir, '3.png')
index = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1
print('index %d: %s' % (index, path_to_image_file))

with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
    attrs = get_attrs(digit_struct_mat_file, index)
    length = len(attrs['label'])
    attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x],
                                                       [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
    min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                min(attrs_top),
                                                max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                max(map(lambda x, y: x + y, attrs_top, attrs_height)))
    center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                    (min_top + max_bottom) / 2.0,
                                    max(max_right - min_left, max_bottom - min_top))
    bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0, 
                                                    center_y - max_side / 2.0, 
                                                    max_side,
                                                    max_side)
    cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                int(round(bbox_top - 0.15 * bbox_height)),
                                                                int(round(bbox_width * 1.3)),
                                                                int(round(bbox_height * 1.3)))
print('min_left=%d, min_top=%d, max_right=%d, max_bottom=%d' % (min_left, min_top, max_right, max_bottom))
print('center_x=%.1f, center_y=%.1f, max_side=%d' % (center_x, center_y, max_side))
print('bbox: left=%.1f, top=%.1f, width=%d, height=%d' % (bbox_left, bbox_top, bbox_width, bbox_height))
print('cropped: left=%d, top=%d, width=%d, height=%d' % (cropped_left, cropped_top, cropped_width, cropped_height))

image = Image.open(path_to_image_file)
plt.figure()
currentAxis = plt.gca()
currentAxis.imshow(image)
currentAxis.add_patch(Rectangle((cropped_left, cropped_top), cropped_width, cropped_height, fill=False, edgecolor='red'))
currentAxis.add_patch(Rectangle((bbox_left, bbox_top), bbox_width, bbox_height, fill=False, edgecolor='green'))
for attr_left, attr_top, attr_width, attr_height in zip(attrs_left, attrs_top, attrs_width, attrs_height):
    currentAxis.add_patch(Rectangle((attr_left, attr_top), attr_width, attr_height, fill=False, edgecolor='white', linestyle='dotted'))
plt.show()
