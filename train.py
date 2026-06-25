from ultralytics import YOLO
import os

def main():
    # 加载预训练的 YOLOv11n 模型
    model = YOLO('yolo11n.pt')

    # 开始基准训练
    results = model.train(
        data='data.yaml',      # 确保仓库根目录下有 data.yaml
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,              # 使用第1块 GPU
        project='Ski_Detection_Baseline',
        name='model1_exp',
        save=True,
        exist_ok=True
    )
    print("Baseline training finished. Results saved to runs/detect/Ski_Detection_Baseline")

if __name__ == '__main__':
    main()
