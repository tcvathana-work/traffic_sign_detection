if __name__ == '__main__':
    from ultralytics import YOLO
    
    save_path = 'results/'

    # Load a model
    model = YOLO("models/yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data="data.yaml", 
        epochs=1, 
        imgsz=640, 
        device="mps", 
        project=save_path  # Use absolute path
    )
