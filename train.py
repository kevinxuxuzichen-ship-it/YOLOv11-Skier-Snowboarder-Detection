from ultralytics import YOLO
import os

def main():
    # Load pre-trained YOLOv11n model
    model = YOLO('yolo11n.pt')

    # Start baseline training
    results = model.train(
        data='data.yaml',      # Ensure data.yaml exists in the root directory of the repository
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,              # Use the 1st GPU
        project='Ski_Detection_Baseline',
        name='model1_exp',
        save=True,
        exist_ok=True
    )
    print("Baseline training finished. Results saved to runs/detect/Ski_Detection_Baseline")

if __name__ == '__main__':
    main()
