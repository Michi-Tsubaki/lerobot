"""このスクリプトは、PushT環境でDiffusion Policyをトレーニングする方法を示しています。

このスクリプトでモデルをトレーニングした後、
examples/2_evaluate_pretrained_policy.py で評価することができます。
"""
import os

from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    print(f"GPU総メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
    print(f"現在の割り当てメモリ: {torch.cuda.memory_allocated(0) / 1e9} GB")
    print(f"現在の予約メモリ: {torch.cuda.memory_reserved(0) / 1e9} GB")
    # トレーニングチェックポイントを保存するディレクトリを作成
    output_directory = Path("outputs/train/my_example_pusht_diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # デバイスを選択
    device = torch.device("cpu")

    # オフライントレーニングのステップ数（この例ではオフラインのみトレーニングを行う）
    # 希望に応じて調整。評価に値するものを得るには5000ステップが必要。
    training_steps = 5000
    log_freq = 1

    # ゼロから始める場合（事前トレーニングされたポリシーではない）、
    # ポリシーを作成する前に2つのことを指定する必要があります：
    #   - 入力/出力の形状：ポリシーを適切にサイズ設定するため
    #   - データセットの統計：入力/出力の正規化と非正規化のため
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # ポリシーは設定クラスで初期化されます。この場合は`DiffusionConfig`。
    # この例では、デフォルトを使用するため、入力/出力の特徴以外の引数は不要です。
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

    # 今、この設定とデータセット統計を使用してポリシーをインスタンス化できます。
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # ポリシーとデータセットのもう一つの相互作用は、delta_timestampsです。
    # 各ポリシーは、入力、出力、および報酬（存在する場合）で異なる数のフレームを期待します。
    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # デバッグ用に設定を出力
    print("Config horizon:", cfg.horizon)
    print("Observation delta indices:", cfg.observation_delta_indices)
    print("Action delta indices:", cfg.action_delta_indices)

    # delta_timestampsを元の設定に戻す
    delta_timestamps = {
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # データセットを作成
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)


    # 次に、オフラインのトレーニング用にオプティマイザとデータローダーを作成します。
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,  # バッチサイズの指定が必要です
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )


    # 勾配スケーラーを初期化
    scaler = GradScaler()

    # トレーニングループを修正
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            # バッチをデバイスに移動
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # 混合精度の計算
            with autocast():
                output_dict = policy.forward(batch)
                loss = output_dict["loss"]
            
            # 勾配スケーリングを使用
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break
    """
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break
    """

    # ポリシーのチェックポイントを保存。
    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()