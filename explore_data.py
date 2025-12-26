import pandas as pd
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    base_dir = 'open_image_dataset/test'
    image_dir = f'{base_dir}/data'
    labels = pd.read_csv(f'{base_dir}/labels/traffic_sign.csv').to_numpy()
    #
    item = labels[0]
    #
    filename = f'{item[0]}.jpg'
    filepath = f'{image_dir}/{filename}'
    # bbox: 4, 5, 6, 7 xmin, ymin, xmax, ymax
    xmin_norm = item[4]
    xmax_norm = item[5]
    ymin_norm = item[6]
    ymax_norm = item[7]

    #
    image = cv2.imread(filepath, cv2.IMREAD_COLOR_RGB)
    height, width = image.shape[0:2]
    # denormalize
    xmin = int(xmin_norm * width)
    ymin = int(ymin_norm * height)
    xmax = int(xmax_norm * width)
    ymax = int(ymax_norm * height)
    #
    color = (0, 255, 0)  # Green in BGR
    thickness = 2
    cv2.rectangle(
        image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=thickness
    )

    plt.imshow(image)
    plt.show()
