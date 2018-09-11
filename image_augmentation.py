import cv2
import numpy as np
import matplotlib.pyplot as plt
import data_handler
import scipy
import csv
import cv2
import numpy as np
import scipy.misc

def flip_horizontal(image, angle):
    return cv2.flip(image, flipCode=1), -angle

def manipulate_brightness(image, min_rand_val = 0.2, max_rand_val = 0.8):
    r = np.random.uniform(min_rand_val, max_rand_val)
    img = image.astype(np.float32)
    img[:,:,:] *= r
    np.clip(img, 0., 255.)
    return img.astype(np.uint8)


def sort_values(val_1, val_2):
    if val_1 > val_2:
        return val_2, val_1
    else:
        return val_1, val_2



def random_shades(image):
    rows = image.shape[0]
    cols = image.shape[1]

    t_1, t_2 = sort_values(np.random.random_sample() * cols, np.random.random_sample() * cols)
    b_1, b_2 = sort_values(np.random.random_sample() * cols, np.random.random_sample() * cols)

    poly = np.asarray([[[t_1, 0], [b_1, rows], [b_2, rows], [t_2, 0]]], dtype=np.int32)

    shade_value = np.random.uniform(0.3,0.8)
    orig_value = 1 - shade_value

    mask = np.copy(image).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))


    return cv2.addWeighted(image.astype(np.int32), orig_value, mask, shade_value, 0).astype(np.uint8)

def get_image(filename):
    return scipy.misc.imread(filename)

def augment_images(data_dir, data_desc_file):
    with open(data_dir + '/' + data_desc_file, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')

        augmented_data = []

        rowNum = 0
        for row in reader:

            if rowNum == 0:
                header = row
            else:
                image = get_image(data_dir + '/' + row[0])
                angle = float(row[3])

                # flip image horizontal
                flip_img, flip_angle = flip_horizontal(image, angle)
                augmented_data.append([flip_img, flip_angle])

                # manipulate brightness
                bright_img = manipulate_brightness(image)
                augmented_data.append([bright_img, angle])

                # apply random shades
                shade_img = random_shades(image)
                augmented_data.append([shade_img, angle])

            rowNum = rowNum + 1

        print(len(augmented_data))

if __name__ == '__main__':
    data_path = './data'
    data_desc_file = 'driving_log.csv'
    augment_images(data_path, data_desc_file)
