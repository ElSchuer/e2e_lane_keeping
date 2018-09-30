import cv2
import numpy as np
import matplotlib.pyplot as plt
import data_handler
import scipy
import csv
import cv2
import numpy as np
import scipy.misc
import os

def flip_horizontal(image, angle):
    return cv2.flip(image, flipCode=1), -angle

def manipulate_brightness(image, min_rand_val = 0.2,  max_rand_val = 0.8):
    r = np.random.uniform(min_rand_val, max_rand_val)
    img = image.astype(np.float32)

    if len(img.shape) == 2:
        img[:, :] *= r
    if len(img.shape) == 3:
        img[:, :, :] *= r

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

def save_augmented_data(augmented_data, path):

    print("Saving Augmented data in " + path)

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '/' + 'augmented_log.csv', mode='a+') as new_log_file:
        s_count = 0
        for dataset in augmented_data:
            new_img = dataset[0]
            new_angle = dataset[1]
            new_img_path = path + '/' +  dataset[2]

            writer = csv.writer(new_log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([new_img_path, '', '', str(new_angle)])

            scipy.misc.imsave(new_img_path, new_img)

            s_count = s_count + 1

            print("Saved " + str(s_count) + " of " + str(len(augmented_data)))



def augment_images(data_dir, data_desc_file):
    augmented_data = []

    new_path = data_dir + '/' + 'augmented_data'

    with open(data_dir + '/' + data_desc_file, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')

        for row in reader:

            print("Processing Row + " + str(row))

            img_name = row[0]
            image = get_image(data_dir + '/' + img_name)
            angle = float(row[3])
            augmented_data.append([image, angle, img_name])

            if angle != 0:
                # flip image horizontal
                flip_img, flip_angle = flip_horizontal(image, angle)
                augmented_data.append([flip_img, flip_angle, img_name[:len(img_name)-4] + "_flip.jpg"])

                # manipulate brightness
                dark_img = manipulate_brightness(image, min_rand_val=0.2, max_rand_val=0.8)
                augmented_data.append([dark_img, angle, img_name[:len(img_name)-4] + "_dark.jpg"])

                bright_img = manipulate_brightness(image, min_rand_val=1.1, max_rand_val=1.9)
                augmented_data.append([bright_img, angle, img_name[:len(img_name)-4] + "_bright.jpg"])

                # apply random shades
                shade_img = random_shades(image)
                augmented_data.append([shade_img, angle, img_name[:len(img_name)-4] + "_shade.jpg"])

                save_augmented_data(augmented_data, new_path)

                augmented_data = []


if __name__ == '__main__':
    data_path = './data/velox_data_path'
    data_desc_file = 'augmented_log.csv'
    augment_images(data_path, data_desc_file)
