import numpy as np
import pandas as pd
import os


base_dataset_dir = 'open_image_dataset'
traffic_sign_id = '/m/01mqdt'

'''
**** Train 2000 images ****
'''
train_files = os.listdir(f'{base_dataset_dir}/train/data')
whole_train_labels = pd.read_csv(f'{base_dataset_dir}/train/labels/detections.csv')
#
existing_train_image_ids = {file.split('.')[0] for file in train_files}
traffic_sign_train_labels = whole_train_labels[
    (whole_train_labels['ImageID'].isin(existing_train_image_ids)) & 
    (whole_train_labels['LabelName'] == traffic_sign_id)
].copy()
traffic_sign_train_labels = traffic_sign_train_labels.drop_duplicates()
traffic_sign_train_labels.to_csv(f'{base_dataset_dir}/train/labels/traffic_sign.csv', index=False)




'''
**** Validate 27 images ****
'''
validation_files = os.listdir(f'{base_dataset_dir}/validation/data')
whole_validation_labels = pd.read_csv(f'{base_dataset_dir}/validation/labels/detections.csv')
#
existing_validate_image_ids = {file.split('.')[0] for file in validation_files}
traffic_sign_validate_labels = whole_validation_labels[
    (whole_validation_labels['ImageID'].isin(existing_validate_image_ids)) &
    (whole_validation_labels['LabelName'] == traffic_sign_id)
]
traffic_sign_validate_labels = traffic_sign_validate_labels.drop_duplicates()
traffic_sign_validate_labels.to_csv(f'{base_dataset_dir}/validation/labels/traffic_sign.csv', index=False)



'''
**** Testing 87 images ****
'''
test_files = os.listdir(f'{base_dataset_dir}/test/data')
whole_test_labels = pd.read_csv(f'{base_dataset_dir}/test/labels/detections.csv')
#
existing_test_image_ids = {file.split('.')[0] for file in test_files}
traffic_sign_test_labels = whole_test_labels[
    (whole_test_labels['ImageID'].isin(existing_test_image_ids)) &
    (whole_test_labels['LabelName'] == traffic_sign_id)
]
traffic_sign_test_labels = traffic_sign_test_labels.drop_duplicates()
traffic_sign_test_labels.to_csv(f'{base_dataset_dir}/test/labels/traffic_sign.csv', index=False)



if __name__ == '__main__':
    print(len(traffic_sign_train_labels))
    print(len(traffic_sign_validate_labels))
    print(len(traffic_sign_test_labels))