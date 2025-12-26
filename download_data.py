import fiftyone as fo
import fiftyone.zoo as foz

# Download only Traffic Sign class
data_source = foz.load_zoo_dataset(
    "open-images-v6",
    splits=("train", "validation", "test"),  # or "validation", "test"
    label_types=["detections"],
    classes=["Traffic sign"],
    max_samples=2000,
)

if __name__ == '__main__':
    print(len(data_source))