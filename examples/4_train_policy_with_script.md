# トレーニングスクリプトのチュートリアル

このチュートリアルでは、トレーニングスクリプトの説明、その使用方法、特にトレーニング実行に必要なすべての設定方法について説明します。

> **注意:** 以下の説明は、CUDA GPUを搭載したマシンでコマンドを実行することを前提としています。GPUがない場合（またはMacを使用している場合）、`--device=cpu`（またはそれぞれ`--device=mps`）を追加できます。ただし、CPUでは実行速度が大幅に遅くなることに注意してください。

## トレーニングスクリプト

LeRobotは[`lerobot/scripts/train.py`](../../lerobot/scripts/train.py)にトレーニングスクリプトを提供しています。概要として、以下のことを行います：

- 以下のステップのための設定を初期化/ロードします。
- データセットをインスタンス化します。
- （オプション）そのデータセットに対応するシミュレーション環境をインスタンス化します。
- ポリシーをインスタンス化します。
- 順伝播、逆伝播、最適化ステップ、および定期的なログ記録、評価（環境上でのポリシーの評価）、チェックポイントを含む標準的なトレーニングループを実行します。

## 設定システムの概要

トレーニングスクリプトでは、メイン関数`train`は`TrainPipelineConfig`オブジェクトを期待します：
```python
# train.py
@parser.wrap()
def train(cfg: TrainPipelineConfig):
```

[`lerobot/configs/train.py`](../../lerobot/configs/train.py)で定義されている`TrainPipelineConfig`を確認できます（これは詳細にコメントが付けられており、すべてのオプションを理解するためのリファレンスとなることを意図しています）

スクリプトを実行する際、コマンドラインからの入力は`@parser.wrap()`デコレータによって解析され、このクラスのインスタンスが自動的に生成されます。内部的には、これは[Draccus](https://github.com/dlwh/draccus)というこの目的のために作られたツールによって行われます。Hydraに慣れている方なら、Draccusも同様に設定ファイル（.json、.yaml）から設定をロードし、さらにコマンドライン入力を通じてそれらの値を上書きできます。Hydraとは異なり、これらの設定は設定ファイルで完全に定義されるのではなく、dataclassを通じてコードで事前定義されています。これにより、より厳密なシリアライズ/デシリアライズ、型付け、およびコード内で設定を直接オブジェクトとして操作することが可能になります（これにより、IDEでのオートコンプリート、定義へのジャンプなどの便利な機能が利用できます）。

簡略化した例を見てみましょう。トレーニング設定には、他の属性の中でも以下のような属性があります：
```python
@dataclass
class TrainPipelineConfig:
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
```
ここで、例えば`DatasetConfig`は以下のように定義されています：
```python
@dataclass
class DatasetConfig:
    repo_id: str
    episodes: list[int] | None = None
    video_backend: str = "pyav"
```

これにより、例えば`TrainPipelineConfig`のインスタンス`cfg`がある場合、`cfg.dataset.repo_id`で`repo_id`の値にアクセスできるという階層関係が作成されます。
コマンドラインから、非常に似た構文`--dataset.repo_id=repo/id`を使用してこの値を指定できます。

デフォルトでは、すべてのフィールドはdataclassで指定されたデフォルト値を取ります。フィールドにデフォルト値がない場合、コマンドラインまたは設定ファイルから指定する必要があります（設定ファイルのパスもコマンドラインで指定します - 詳細は後述）。上記の例では、`dataset`フィールドにデフォルト値がないため、指定する必要があります。

## CLIから値を指定する

[Diffusion Policy](../../lerobot/common/policies/diffusion)を[pusht](https://huggingface.co/datasets/lerobot/pusht)データセットで訓練し、評価のために[gym_pusht](https://github.com/huggingface/gym-pusht)環境を使用したいとしましょう。そのためのコマンドは以下のようになります：
```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=lerobot/pusht \
    --policy.type=diffusion \
    --env.type=pusht
```

これを分解して説明しましょう：
- データセットを指定するには、`DatasetConfig`で唯一必須の引数であるハブ上の`repo_id`を指定するだけで良いです。残りのフィールドにはデフォルト値があり、この場合はそれらで問題ないため、単に`--dataset.repo_id=lerobot/pusht`オプションを追加するだけです。
- ポリシーを指定するには、`--policy`に`.type`を付けてdiffusionポリシーを選択するだけです。ここで、`.type`は`draccus.ChoiceRegistry`を継承し、`register_subclass()`メソッドでデコレートされた設定クラスを選択できる特別な引数です。この機能の詳細な説明については、この[Draccusデモ](https://github.com/dlwh/draccus?tab=readme-ov-file#more-flexible-configuration-with-choice-types)をご覧ください。私たちのコードでは、この仕組みを主にポリシー、環境、ロボット、およびオプティマイザなどの他のコンポーネントの選択に使用しています。選択可能なポリシーは[lerobot/common/policies](../../lerobot/common/policies)にあります。
- 同様に、`--env.type=pusht`で環境を選択します。利用可能な環境設定は[`lerobot/common/envs/configs.py`](../../lerobot/common/envs/configs.py)にあります。

別の例を見てみましょう。[ACT](../../lerobot/common/policies/act)を[lerobot/aloha_sim_insertion_human](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human)で訓練し、評価のために[gym-aloha](https://github.com/huggingface/gym-aloha)環境を使用していたとします：
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --output_dir=outputs/train/act_aloha_insertion
```
> `--output_dir`を追加して、この実行からの出力（チェックポイント、トレーニング状態、設定など）を書き込む場所を明示的に指定したことに注意してください。これは必須ではなく、指定しない場合は、現在の日付と時刻、env.type、policy.typeから生成されたデフォルトのディレクトリが作成されます。これは通常`outputs/train/2025-01-24/16-10-05_aloha_act`のようになります。

今度は別のタスクでalohaの別のポリシーを訓練したいとします。データセットを変更して、代わりに[lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)を使用します。もちろん、このタスクに合わせて環境のタスクも変更する必要があります。
[`AlohaEnv`](../../lerobot/common/envs/configs.py)設定を見ると、タスクはデフォルトで`"AlohaInsertion-v0"`で、これは上記のコマンドで訓練したタスクに対応します。[gym-aloha](https://github.com/huggingface/gym-aloha?tab=readme-ov-file#description)環境には、訓練したい別のタスクに対応する`AlohaTransferCube-v0`タスクもあります。これらをまとめると、以下のコマンドで新しいポリシーをこの異なるタスクで訓練できます：
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --output_dir=outputs/train/act_aloha_transfer
```

## 設定ファイルからのロード

さて、上記の実行を再現したいとします。その実行では、使用した`TrainPipelineConfig`インスタンスをシリアライズした`train_config.json`ファイルがチェックポイントに生成されています：
```json
{
    "dataset": {
        "repo_id": "lerobot/aloha_sim_transfer_cube_human",
        "episodes": null,
        ...
    },
    "env": {
        "type": "aloha",
        "task": "AlohaTransferCube-v0",
        "fps": 50,
        ...
    },
    "policy": {
        "type": "act",
        "n_obs_steps": 1,
        ...
    },
    ...
}
```

以下のように、このファイルから設定値を簡単にロードできます：
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model/ \
    --output_dir=outputs/train/act_aloha_transfer_2
```
`--config_path`は、ローカルの設定ファイルから設定を初期化できる特別な引数です。`train_config.json`を含むディレクトリを指定するか、設定ファイル自体を直接指定できます。

Hydraと同様に、必要に応じてCLIでパラメータを上書きすることもできます：
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model/ \
    --output_dir=outputs/train/act_aloha_transfer_2
    --policy.n_action_steps=80
```
> 注意：一般的に`--output_dir`は必須ではありませんが、この場合は`train_config.json`から値を取得するため（値は`outputs/train/act_aloha_transfer`）、指定する必要があります。以前の実行のチェックポイントを誤って削除することを防ぐため、既存のディレクトリに書き込もうとするとエラーが発生します。これは実行を再開する場合には当てはまりません（次に学びます）。

`--config_path`は、`train_config.json`ファイルを含むハブ上のリポジトリのrepo_idも受け入れることができます。例えば：
```bash
python lerobot/scripts/train.py --config_path=lerobot/diffusion_pusht
```
を実行すると、[lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht)の訓練に使用されたのと同じ設定で訓練実行が開始されます。

## トレーニングの再開

クラッシュや中断などの理由で訓練実行を再開できることは重要です。ここでその方法を説明します。

前回の実行のコマンドを再利用し、いくつかのオプションを追加してみましょう：
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --log_freq=25 \
    --save_freq=100 \
    --output_dir=outputs/train/run_resumption
```

ここでは、再開のデモンストレーションができるように、ログ頻度とチェックポイント頻度を低い数値に設定しました。（ハードウェアによりますが）1分以内にログが表示され、最初のチェックポイントが作成されるはずです。最初のチェックポイントが作成されるのを待ちます。ターミナルに以下のような行が表示されるはずです：
```
INFO 2025-01-24 16:10:56 ts/train.py:263 Checkpoint policy after step 100
```
ここで、プロセスを強制終了して（`ctrl`+`c`を押して）クラッシュをシミュレートしましょう。その後、以下のコマンドで利用可能な最後のチェックポイントから実行

＝＝＝＝＝＝


This tutorial will explain the training script, how to use it, and particularly how to configure everything needed for the training run.
> **Note:** The following assume you're running these commands on a machine equipped with a cuda GPU. If you don't have one (or if you're using a Mac), you can add `--device=cpu` (`--device=mps` respectively). However, be advised that the code executes much slower on cpu.


## The training script

LeRobot offers a training script at [`lerobot/scripts/train.py`](../../lerobot/scripts/train.py). At a high level it does the following:

- Initialize/load a configuration for the following steps using.
- Instantiates a dataset.
- (Optional) Instantiates a simulation environment corresponding to that dataset.
- Instantiates a policy.
- Runs a standard training loop with forward pass, backward pass, optimization step, and occasional logging, evaluation (of the policy on the environment), and checkpointing.

## Overview of the configuration system

In the training script, the main function `train` expects a `TrainPipelineConfig` object:
```python
# train.py
@parser.wrap()
def train(cfg: TrainPipelineConfig):
```

You can inspect the `TrainPipelineConfig` defined in [`lerobot/configs/train.py`](../../lerobot/configs/train.py) (which is heavily commented and meant to be a reference to understand any option)

When running the script, inputs for the command line are parsed thanks to the `@parser.wrap()` decorator and an instance of this class is automatically generated. Under the hood, this is done with [Draccus](https://github.com/dlwh/draccus) which is a tool dedicated for this purpose. If you're familiar with Hydra, Draccus can similarly load configurations from config files (.json, .yaml) and also override their values through command line inputs. Unlike Hydra, these configurations are pre-defined in the code through dataclasses rather than being defined entirely in config files. This allows for more rigorous serialization/deserialization, typing, and to manipulate configuration as objects directly in the code and not as dictionaries or namespaces (which enables nice features in an IDE such as autocomplete, jump-to-def, etc.)

Let's have a look at a simplified example. Amongst other attributes, the training config has the following attributes:
```python
@dataclass
class TrainPipelineConfig:
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
```
in which `DatasetConfig` for example is defined as such:
```python
@dataclass
class DatasetConfig:
    repo_id: str
    episodes: list[int] | None = None
    video_backend: str = "pyav"
```

This creates a hierarchical relationship where, for example assuming we have a `cfg` instance of `TrainPipelineConfig`, we can access the `repo_id` value with `cfg.dataset.repo_id`.
From the command line, we can specify this value with using a very similar syntax `--dataset.repo_id=repo/id`.

By default, every field takes its default value specified in the dataclass. If a field doesn't have a default value, it needs to be specified either from the command line or from a config file – which path is also given in the command line (more in this below). In the example above, the `dataset` field doesn't have a default value which means it must be specified.


## Specifying values from the CLI

Let's say that we want to train [Diffusion Policy](../../lerobot/common/policies/diffusion) on the [pusht](https://huggingface.co/datasets/lerobot/pusht) dataset, using the [gym_pusht](https://github.com/huggingface/gym-pusht) environment for evaluation. The command to do so would look like this:
```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=lerobot/pusht \
    --policy.type=diffusion \
    --env.type=pusht
```

Let's break this down:
- To specify the dataset, we just need to specify its `repo_id` on the hub which is the only required argument in the `DatasetConfig`. The rest of the fields have default values and in this case we are fine with those so we can just add the option `--dataset.repo_id=lerobot/pusht`.
- To specify the policy, we can just select diffusion policy using `--policy` appended with `.type`. Here, `.type` is a special argument which allows us to select config classes inheriting from `draccus.ChoiceRegistry` and that have been decorated with the `register_subclass()` method. To have a better explanation of this feature, have a look at this [Draccus demo](https://github.com/dlwh/draccus?tab=readme-ov-file#more-flexible-configuration-with-choice-types). In our code, we use this mechanism mainly to select policies, environments, robots, and some other components like optimizers. The policies available to select are located in [lerobot/common/policies](../../lerobot/common/policies)
- Similarly, we select the environment with `--env.type=pusht`. The different environment configs are available in [`lerobot/common/envs/configs.py`](../../lerobot/common/envs/configs.py)

Let's see another example. Let's say you've been training [ACT](../../lerobot/common/policies/act) on [lerobot/aloha_sim_insertion_human](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) using the [gym-aloha](https://github.com/huggingface/gym-aloha) environment for evaluation with:
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --output_dir=outputs/train/act_aloha_insertion
```
> Notice we added `--output_dir` to explicitly tell where to write outputs from this run (checkpoints, training state, configs etc.). This is not mandatory and if you don't specify it, a default directory will be created from the current date and time, env.type and policy.type. This will typically look like `outputs/train/2025-01-24/16-10-05_aloha_act`.

We now want to train a different policy for aloha on another task. We'll change the dataset and use [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) instead. Of course, we also need to change the task of the environment as well to match this other task.
Looking at the [`AlohaEnv`](../../lerobot/common/envs/configs.py) config, the task is `"AlohaInsertion-v0"` by default, which corresponds to the task we trained on in the command above. The [gym-aloha](https://github.com/huggingface/gym-aloha?tab=readme-ov-file#description) environment also has the `AlohaTransferCube-v0` task which corresponds to this other task we want to train on. Putting this together, we can train this new policy on this different task using:
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --output_dir=outputs/train/act_aloha_transfer
```

## Loading from a config file

Now, let's assume that we want to reproduce the run just above. That run has produced a `train_config.json` file in its checkpoints, which serializes the `TrainPipelineConfig` instance it used:
```json
{
    "dataset": {
        "repo_id": "lerobot/aloha_sim_transfer_cube_human",
        "episodes": null,
        ...
    },
    "env": {
        "type": "aloha",
        "task": "AlohaTransferCube-v0",
        "fps": 50,
        ...
    },
    "policy": {
        "type": "act",
        "n_obs_steps": 1,
        ...
    },
    ...
}
```

We can then simply load the config values from this file using:
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model/ \
    --output_dir=outputs/train/act_aloha_transfer_2
```
`--config_path` is also a special argument which allows to initialize the config from a local config file. It can point to a directory that contains `train_config.json` or to the config file itself directly.

Similarly to Hydra, we can still override some parameters in the CLI if we want to, e.g.:
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model/ \
    --output_dir=outputs/train/act_aloha_transfer_2
    --policy.n_action_steps=80
```
> Note: While `--output_dir` is not required in general, in this case we need to specify it since it will otherwise take the value from the `train_config.json` (which is `outputs/train/act_aloha_transfer`). In order to prevent accidental deletion of previous run checkpoints, we raise an error if you're trying to write in an existing directory. This is not the case when resuming a run, which is what you'll learn next.

`--config_path` can also accept the repo_id of a repo on the hub that contains a `train_config.json` file, e.g. running:
```bash
python lerobot/scripts/train.py --config_path=lerobot/diffusion_pusht
```
will start a training run with the same configuration used for training [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht)


## Resume training

Being able to resume a training run is important in case it crashed or aborted for any reason. We'll demonstrate how to that here.

Let's reuse the command from the previous run and add a few more options:
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --log_freq=25 \
    --save_freq=100 \
    --output_dir=outputs/train/run_resumption
```

Here we've taken care to set up the log frequency and checkpointing frequency to low numbers so we can showcase resumption. You should be able to see some logging and have a first checkpoint within 1 minute (depending on hardware). Wait for the first checkpoint to happen, you should see a line that looks like this in your terminal:
```
INFO 2025-01-24 16:10:56 ts/train.py:263 Checkpoint policy after step 100
```
Now let's simulate a crash by killing the process (hit `ctrl`+`c`). We can then simply resume this run from the last checkpoint available with:
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/run_resumption/checkpoints/last/pretrained_model/ \
    --resume=true
```
You should see from the logging that your training picks up from where it left off.

Another reason for which you might want to resume a run is simply to extend training and add more training steps. The number of training steps is set by the option `--offline.steps`, which is 100 000 by default.
You could double the number of steps of the previous run with:
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/run_resumption/checkpoints/last/pretrained_model/ \
    --resume=true \
    --offline.steps=200000
```

## Outputs of a run
In the output directory, there will be a folder called `checkpoints` with the following structure:
```bash
outputs/train/run_resumption/checkpoints
├── 000100  # checkpoint_dir for training step 100
│   ├── pretrained_model
│   │   ├── config.json  # pretrained policy config
│   │   ├── model.safetensors  # model weights
│   │   ├── train_config.json  # train config
│   │   └── README.md  # model card
│   └── training_state.pth  # optimizer/scheduler/rng state and training step
├── 000200
└── last -> 000200  # symlink to the last available checkpoint
```

## Fine-tuning a pre-trained policy

In addition to the features currently in Draccus, we've added a special `.path` argument for the policy, which allows to load a policy as you would with `PreTrainedPolicy.from_pretrained()`. In that case, `path` can be a local directory that contains a checkpoint or a repo_id pointing to a pretrained policy on the hub.

For example, we could fine-tune a [policy pre-trained on the aloha transfer task](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human) on the aloha insertion task. We can achieve this with:
```bash
python lerobot/scripts/train.py \
    --policy.path=lerobot/act_aloha_sim_transfer_cube_human \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0
```

When doing so, keep in mind that the features of the fine-tuning dataset would have to match the input/output features of the pretrained policy.

## Typical logs and metrics

When you start the training process, you will first see your full configuration being printed in the terminal. You can check it to make sure that you configured your run correctly. The final configuration will also be saved with the checkpoint.

After that, you will see training log like this one:
```
INFO 2024-08-14 13:35:12 ts/train.py:192 step:0 smpl:64 ep:1 epch:0.00 loss:1.112 grdn:15.387 lr:2.0e-07 updt_s:1.738 data_s:4.774
```
or evaluation log:
```
INFO 2024-08-14 13:38:45 ts/train.py:226 step:100 smpl:6K ep:52 epch:0.25 ∑rwrd:20.693 success:0.0% eval_s:120.266
```

These logs will also be saved in wandb if `wandb.enable` is set to `true`. Here are the meaning of some abbreviations:
- `smpl`: number of samples seen during training.
- `ep`: number of episodes seen during training. An episode contains multiple samples in a complete manipulation task.
- `epch`: number of time all unique samples are seen (epoch).
- `grdn`: gradient norm.
- `∑rwrd`: compute the sum of rewards in every evaluation episode and then take an average of them.
- `success`: average success rate of eval episodes. Reward and success are usually different except for the sparsing reward setting, where reward=1 only when the task is completed successfully.
- `eval_s`: time to evaluate the policy in the environment, in second.
- `updt_s`: time to update the network parameters, in second.
- `data_s`: time to load a batch of data, in second.

Some metrics are useful for initial performance profiling. For example, if you find the current GPU utilization is low via the `nvidia-smi` command and `data_s` sometimes is too high, you may need to modify batch size or number of dataloading workers to accelerate dataloading. We also recommend [pytorch profiler](https://github.com/huggingface/lerobot?tab=readme-ov-file#improve-your-code-with-profiling) for detailed performance probing.

## In short

We'll summarize here the main use cases to remember from this tutorial.

#### Train a policy from scratch – CLI
```bash
python lerobot/scripts/train.py \
    --policy.type=act \  # <- select 'act' policy
    --env.type=pusht \  # <- select 'pusht' environment
    --dataset.repo_id=lerobot/pusht  # <- train on this dataset
```

#### Train a policy from scratch - config file + CLI
```bash
python lerobot/scripts/train.py \
    --config_path=path/to/pretrained_model \  # <- can also be a repo_id
    --policy.n_action_steps=80  # <- you may still override values
```

#### Resume/continue a training run
```bash
python lerobot/scripts/train.py \
    --config_path=checkpoint/pretrained_model/ \
    --resume=true \
    --offline.steps=200000  # <- you can change some training parameters
```

#### Fine-tuning
```bash
python lerobot/scripts/train.py \
    --policy.path=lerobot/act_aloha_sim_transfer_cube_human \  # <- can also be a local path to a checkpoint
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0
```

---

Now that you know the basics of how to train a policy, you might want to know how to apply this knowledge to actual robots, or how to record your own datasets and train policies on your specific task?
If that's the case, head over to the next tutorial [`7_get_started_with_real_robot.md`](./7_get_started_with_real_robot.md).

Or in the meantime, happy training! 🤗
