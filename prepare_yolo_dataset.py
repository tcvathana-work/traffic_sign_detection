import pandas as pd
from pathlib import Path
import shutil

if __name__ == '__main__':
    modes = ['train', 'validation', 'test']
    
    for mode in modes:
        # Copy images, remove if exists and recreate
        image_src_dir = f'open_image_dataset/{mode}/data'
        image_dst_dir = f'yolo_dataset/images/{mode}'
        if Path(image_dst_dir).exists():
            shutil.rmtree(image_dst_dir)
        #
        shutil.copytree(src=image_src_dir, dst=image_dst_dir)
        
        # Create labels directory, remove if exists and recreate
        label_output_dir = f'yolo_dataset/labels/{mode}'
        if Path(label_output_dir).exists():
            shutil.rmtree(label_output_dir)
        # Create labels directory
        Path(label_output_dir).mkdir(parents=True, exist_ok=True)
        
        #
        annotations = {}
        #
        labels = pd.read_csv(f'open_image_dataset/{mode}/labels/traffic_sign.csv').to_numpy()
        unique_ids = set(labels[:, 0])
        class_id = 0
        #
        for i, label in enumerate(labels):
            filename = label[0]
            
            xmin_norm = label[4]
            xmax_norm = label[5]
            ymin_norm = label[6]
            ymax_norm = label[7]
            #
            xcenter_norm = (xmax_norm + xmin_norm) / 2
            ycenter_norm = (ymax_norm + ymin_norm) / 2
            width_norm = xmax_norm - xmin_norm
            height_norm = ymax_norm - ymin_norm    
            # <class_id> <x_center> <y_center> <width> <height>
            bbox = f'{class_id} {xcenter_norm:4f} {ycenter_norm:4f} {width_norm:4f} {height_norm:4f}\n'
            with open(f'{label_output_dir}/{filename}.txt', "a") as file:
                file.write(bbox)
                file.close()

    
    
    