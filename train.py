from pathlib import Path

import torch

from ultralytics import YOLO


def main():
    root_dir = Path(__file__).resolve().parent

    # 自定义实验名称 没加exp默认是该模块的第一次训练 exp1表示该模块训练的第一次 exp2表示修改了该模块训练的第二次
    run_name = "2160-yolo26-drcc-exp1-e200-b16-s42"

    # 训练结果固定保存到项目根目录 outputs
    out_dir = (root_dir / "outputs" / "no-pretrained").resolve()
    # out_dir = root_dir / "outputs" / "pretrained"

    # 检查上次训练是否训练完成
    run_dir = (out_dir / run_name).resolve()
    last_ckpt = run_dir / "weights" / "last.pt"

    # 是否继续训练，默认为false
    resume = False

    # 固定随机种子
    seed = 42

    # 模型配置
    model_cfg = root_dir / "yolo26n-drcc.yaml"

    # 预训练权重
    pretrained_ckpt = root_dir / "yolo26n.pt"

    device = 0 if torch.cuda.is_available() else "cpu"

    if last_ckpt.exists():
        model = YOLO(str(last_ckpt))
        resume = str(last_ckpt)
        pretrained = False
    else:
        model = YOLO(str(model_cfg))
        resume = False
        pretrained = str(pretrained_ckpt)

    # 开始训练
    model.train(
        data=str(root_dir / "datasets" / "tt100k_aug_2160" / "TT100K.yaml"),  # 数据集配置
        epochs=200,
        imgsz=640,
        batch=16,
        device=device,
        resume=resume,
        # pretrained=pretrained,
        pretrained=False,
        workers=2,
        seed=seed,
        project=str(out_dir),
        name=run_name,
        # 恢复训练时显式覆盖 checkpoint 中保存的旧路径，避免 Windows 路径转义或工作目录变化。
        save_dir=str(run_dir),
        exist_ok=True,
        patience=50,
    )


if __name__ == "__main__":
    main()
