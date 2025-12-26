### 1. The `download_data.py` will download to `~/fiftyone/open-images-v6`
- Copy all inside `~/fiftyone/open-images-v6/` to this `open_image_dataset/`

---

### 2. The `data_cleansing.py` produce below file
- `open_image_dataset/train/labels/filtered_traffic_sign.csv`


### 2.1 The `explore_data.py` is used to explore the `labels` produced

---

### 3. Run `prepare_yolo_dataset.py` and produces below `yolo_dataset/` 
- `yolo_dataset/images/train`
- `yolo_dataset/labels/train`

