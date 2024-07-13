# 音轨分离通用训练代码

用于训练音轨分离模型的代码库。该库基于[kuielab代码](https://github.com/kuielab/sdx23/tree/mdx_AB/my_submission/src)用于[SDX23挑战](https://github.com/kuielab/sdx23/tree/mdx_AB/my_submission/src)。本代码库的主要目的是创建易于修改的训练代码，以便进行实验。由[MVSep.com](https://mvsep.com)提供。

## 模型

可以通过`--model_type`参数选择模型。

可供训练的模型：
* 基于[KUIELab TFC TDF v3架构](https://github.com/kuielab/sdx23/)的MDX23C。键值：`mdx23c`。
* Demucs4HT [[论文](https://arxiv.org/abs/2211.08553)]。键值：`htdemucs`。
* 基于[Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch)的VitLarge23。键值：`segm_models`。
* 基于[TorchSeg模块](https://github.com/qubvel/segmentation_models.pytorch)的TorchSeg。键值：`torchseg`。
* Band Split RoFormer [[论文](https://arxiv.org/abs/2309.02612), [代码库](https://github.com/lucidrains/BS-RoFormer)]。键值：`bs_roformer`。
* Mel-Band RoFormer [[论文](https://arxiv.org/abs/2310.01809), [代码库](https://github.com/lucidrains/BS-RoFormer)]。键值：`mel_band_roformer`。
* Swin Upernet [[论文](https://arxiv.org/abs/2103.14030)]。键值：`swin_upernet`。
* BandIt Plus [[论文](https://arxiv.org/abs/2309.02539), [代码库](https://github.com/karnwatcharasupat/bandit)]。键值：`bandit`。
* SCNet [[论文](https://arxiv.org/abs/2401.13276), [官方代码库](https://github.com/starrytong/SCNet), [非官方代码库](https://github.com/amanteur/SCNet-PyTorch)]。键值：`scnet`。

- **注意1**：对于`segm_models`，有很多不同的编码器可用。[点击此处查看](https://github.com/qubvel/segmentation_models.pytorch#encoders-)。
- **注意2**：感谢[@lucidrains](https://github.com/lucidrains)基于论文重现了RoFormer模型。
- **注意3**：`torchseg`提供了访问`timm`模块中800多个编码器的功能，类似于`segm_models`。

## 如何训练

要训练模型，你需要：

1. 使用键值`--model_type`选择模型类型。可能的值：`mdx23c`，`htdemucs`，`segm_models`，`mel_band_roformer`，`bs_roformer`。
2. 选择模型配置的路径`--config_path` `<配置路径>`。你可以在[configs文件夹](configs/)中找到配置示例。前缀`config_musdb18_`是[MUSDB18数据集](https://sigsep.github.io/datasets/musdb.html)的示例。
3. 如果你有相同模型或相似模型的检查点，可以使用：`--start_check_point` `<权重路径>`
4. 选择保存训练结果的路径`--results_path` `<结果文件夹路径>`

#### 示例
```bash
python train.py \ 
    --model_type mel_band_roformer \ 
    --config_path configs/config_mel_band_roformer_vocals.yaml \
    --start_check_point results/model.ckpt \
    --results_path results/ \
    --data_path 'datasets/dataset1' 'datasets/dataset2' \
    --valid_path datasets/musdb18hq/test \
    --num_workers 4 \
    --device_ids 0
```

所有可用的训练参数可以在[这里](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/train.py#L109)找到。

## 如何推理

#### 示例

```bash
python inference.py \  
    --model_type mdx23c \
    --config_path configs/config_mdx23c_musdb18.yaml \
    --start_check_point results/last_mdx23c.ckpt \
    --input_folder input/wavs/ \
    --store_dir separation_results/
```

所有可用的推理参数可以在[这里](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/inference.py#L54)找到。

## 有用的说明

* 配置中的所有批处理大小均调整为单个NVIDIA A6000 48GB显存使用。如果你的显存较少，请在模型配置中相应调整`training.batch_size`和`training.gradient_accumulation_steps`。
* 通常，使用旧权重开始训练更好，即使形状不完全匹配。代码支持加载不完全相同的模型权重（但必须具有相同的架构）。这样训练速度会更快。

## 代码描述

* `configs/config_*.yaml` - 模型的配置文件
* `models/*` - 可用于训练和推理的模型集合
* `dataset.py` - 用于创建训练样本的数据集
* `inference.py` - 处理音乐文件夹并分离它们
* `train.py` - 主训练代码
* `utils.py` - 训练/验证中使用的常见函数
* `valid.py` - 使用指标验证模型

## 预训练模型

如果你训练了一些优秀的模型，请分享。你可以在[此问题](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/1)中发布配置和模型权重。

### 人声模型
|                            模型类型                            |  乐器  |   指标 (SDR)   | 配置 | 权重 |
|:----------------------------------------------------------------:|:-------------:|:-----------------:|:-----:|:-----:|
|                              MDX23C                              | 人声 / 其他 | 人声SDR: 10.17 | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_mdx23c.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt) |
|                   HTDemucs4 (MVSep微调)                    | 人声 / 其他 | 人声SDR: 8.78  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_htdemucs.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt) |
|                     Segm Models (VitLarge23)                     | 人声 / 其他 | 人声SDR: 9.77  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_segm_models.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt) |
|                        Mel Band RoFormer                         | 人声 (*) / 其他 | 人声SDR: 8.42  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_mel_band_roformer.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mel_band_roformer_sdr_8.42.ckpt) |
|                           Swin Upernet                           | 人声 / 其他 | 人声SDR: 7.57  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.2/config_vocals_swin_upernet.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.2/model_swin_upernet_ep_56_sdr_10.6703.ckpt) |
| BS Roformer ([viperx](https://github.com/playdasegunda) 版) | 人声 / 其他 | 人声SDR: 10.87 | [配置](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml) | [权重](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755

.ckpt) |
|                      BandIt+ ([配置](https://github.com/karnwatcharasupat/bandit/tree/main/configs))                        | 人声 | 人声SDR: 8.50  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_banditplus.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_banditplus_sdr_8.50.ckpt) |
|              SCNet ([amanteur](https://github.com/amanteur))             | 人声 | 人声SDR: 10.60 | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.3/config_vocals_scnet.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.3/model_vocals_scnet.ckpt) |

### 贝斯模型
|                            模型类型                            |  乐器  |   指标 (SDR)   | 配置 | 权重 |
|:----------------------------------------------------------------:|:-------------:|:-----------------:|:-----:|:-----:|
|                              MDX23C                              | 贝斯 / 其他 | 贝斯SDR: 8.64 | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_bass_mdx23c.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_bass_mdx23c_sdr_8.64.ckpt) |
|                   HTDemucs4 (MVSep微调)                    | 贝斯 / 其他 | 贝斯SDR: 7.68  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_bass_htdemucs.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_bass_htdemucs_sdr_7.68.ckpt) |
|                     Segm Models (VitLarge23)                     | 贝斯 / 其他 | 贝斯SDR: 9.05  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_bass_segm_models.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_bass_segm_models_sdr_9.05.ckpt) |
|                        Mel Band RoFormer                         | 贝斯 (*) / 其他 | 贝斯SDR: 6.68  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_bass_mel_band_roformer.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_bass_mel_band_roformer_sdr_6.68.ckpt) |

### 鼓模型
|                            模型类型                            |  乐器  |   指标 (SDR)   | 配置 | 权重 |
|:----------------------------------------------------------------:|:-------------:|:-----------------:|:-----:|:-----:|
|                              MDX23C                              | 鼓 / 其他 | 鼓SDR: 8.73 | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_drums_mdx23c.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_drums_mdx23c_sdr_8.73.ckpt) |
|                   HTDemucs4 (MVSep微调)                    | 鼓 / 其他 | 鼓SDR: 7.87  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_drums_htdemucs.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_drums_htdemucs_sdr_7.87.ckpt) |
|                     Segm Models (VitLarge23)                     | 鼓 / 其他 | 鼓SDR: 9.47  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_drums_segm_models.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_drums_segm_models_sdr_9.47.ckpt) |
|                        Mel Band RoFormer                         | 鼓 (*) / 其他 | 鼓SDR: 6.69  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_drums_mel_band_roformer.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_drums_mel_band_roformer_sdr_6.69.ckpt) |

### 其他模型
|                            模型类型                            |  乐器  |   指标 (SDR)   | 配置 | 权重 |
|:----------------------------------------------------------------:|:-------------:|:-----------------:|:-----:|:-----:|
|                              MDX23C                              | 其他 / 其他 | 其他SDR: 7.47 | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_other_mdx23c.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_other_mdx23c_sdr_7.47.ckpt) |
|                   HTDemucs4 (MVSep微调)                    | 其他 / 其他 | 其他SDR: 6.82  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_other_htdemucs.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_other_htdemucs_sdr_6.82.ckpt) |
|                     Segm Models (VitLarge23)                     | 其他 / 其他 | 其他SDR: 9.37  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_other_segm_models.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_other_segm_models_sdr_9.37.ckpt) |
|                        Mel Band RoFormer                         | 其他 (*) / 其他 | 其他SDR: 6.56  | [配置](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_other_mel_band_roformer.yaml) | [权重](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_other_mel_band_roformer_sdr_6.56.ckpt) |

> **注意**：
> - 这些模型的预训练权重可以帮助加速你的训练过程，并且可能提升模型的性能。
> - 不同模型和配置文件的性能可能会有所差异。请根据你的需求选择适合的模型和配置文件。

## 安装和依赖项

请确保你已安装以下依赖项：

```bash
pip install torch torchvision torchaudio
pip install numpy scipy pandas tqdm librosa scikit-learn
pip install segmentation_models_pytorch
pip install timm
```

根据你的需要，你可能还需要安装其他依赖项。详细信息可以查看每个模型的官方文档。

## 贡献和反馈

如果你对本项目有任何问题或建议，欢迎提交[Issues](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues)或Pull Requests。

## 许可证

本项目使用MIT许可证，详情请查看LICENSE文件。

## 联系

有关更多信息，请访问[MVSep.com](https://mvsep.com)或联系项目维护者。
