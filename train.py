from pathlib import Path

import torch

from ultralytics import YOLO

  
def main():
    root_dir = Path(__file__).resolve().parent

    # 自定义实验名称
    run_name = "540-yolo26-efcm-e200-b16-s42"

    # 训练结果固定保存到项目根目录 outputs
    out_dir = root_dir / "outputs" / "no-pretrained"
    # out_dir = root_dir / "outputs" / "pretrained"

    # 检查上次训练是否训练完成
    last_ckpt = Path(out_dir) / run_name / "weights" / "last.pt"

    # 是否继续训练，默认为false
    resume = False

    # 固定随机种子
    seed = 42

    # 模型配置
    model_cfg = "./yolo26n-psicconv.yaml"

    # 预训练权重
    pretrained_ckpt = "yolo26n.pt"

    device = 0 if torch.cuda.is_available() else "cpu"

    if last_ckpt.exists():
        model = YOLO(str(last_ckpt))
        resume=True
        pretrained=False
    else:
        model = YOLO(model_cfg)
        pretrained=pretrained_ckpt

    # 开始训练
    model.train(
        data="./datasets/tt100k_aug/TT100K.yaml",  # 数据集配置
        epochs=200,
        imgsz=640,
        batch=16,
        device=device,
        resume=resume,
        # pretrained=pretrained,
        pretrained=False,
        workers=2,
        seed=seed,
        project=out_dir,
        name=run_name,
        patience=50,
    )


if __name__ == "__main__":
    main()
