# HiFi-GAN Vocoder for Russian TTS

Implementation of HiFi-GAN vocoder trained on the RUSLAN dataset for Russian text-to-speech synthesis.

## Links

- **Detailed Report**: [REPORT.md](./REPORT.md) — подробное описание архитектуры, лоссов и экспериментов
- **Quality Analysis**: [analysis.ipynb](./analysis.ipynb) — анализ качества вокодера (waveforms, spectrograms, Mel L1)
- **CometML Training Logs**: [https://www.comet.com/krugd/hifigan-russian-tts](https://www.comet.com/krugd/hifigan-russian-tts/444cc4e3c69c439cbd2f2186e108e03c?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step)
- **Model Weights (Google Drive)**: [generator_best.pt](https://drive.google.com/file/d/1LTl3m5ZScfefGvQUXTBAYfBeoSKhDZR2/view?usp=sharing)

## Overview

This project implements the HiFi-GAN V1 vocoder architecture from the paper:
> **HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis**  
> Jungil Kong, Jaehyeon Kim, Jaekyoung Bae  
> [arXiv:2010.05646](https://arxiv.org/abs/2010.05646)

### Key Features

- **High-quality audio synthesis** from mel-spectrograms
- **Multi-Period Discriminator (MPD)** for capturing periodic patterns in speech
- **Multi-Scale Discriminator (MSD)** for modeling consecutive patterns
- **Multi-Receptive Field Fusion (MRF)** for diverse receptive field patterns in the generator
- **Training on RUSLAN** - Russian language speech dataset (22050 Hz)
- **CometML logging** for experiment tracking

## Project Structure

```
HW3_TTS/
├── src/
│   ├── configs/           # Hydra configuration files
│   ├── datasets/          # RUSLAN and CustomDir datasets
│   ├── model/             # HiFi-GAN Generator and Discriminators
│   ├── loss/              # Loss functions (Adversarial, Mel, Feature Matching)
│   ├── transforms/        # Mel-spectrogram transforms
│   ├── trainer/           # Training logic
│   ├── logger/            # CometML logger
│   └── utils/             # Utility functions
├── data/                  # Dataset directory
├── models_weights/        # Pre-trained model weights
├── checkpoints/           # Training checkpoints
├── train.py               # Training script
├── synthesize.py          # Synthesis script
├── download_ruslan.py     # Dataset download script
├── download_weights.py    # Model weights download script
├── demo.ipynb             # Demo notebook (Google Colab compatible)
├── analysis.ipynb         # Quality analysis notebook
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── REPORT.md              # Detailed implementation report
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/KrugD/HW3_TTS.git
cd HW3_TTS
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up CometML (for training)

Create a `.env` file in the project root:

```bash
COMET_API_KEY=your_api_key_here
```

## Dataset

### Download RUSLAN Dataset

```bash
python download_ruslan.py
```

This will download and extract the RUSLAN dataset to `data/ruslan/RUSLAN/`.

RUSLAN dataset specifications:
- Sample rate: 44100 Hz (resampled to 22050 Hz during training)
- Single speaker
- Russian language

## Training

### Basic Training

```bash
python train.py
```

### Training HiFi-GAN V1

```bash
python train.py model=hifigan_v1
```

### Training with Custom Parameters

```bash
python train.py \
    trainer.n_epochs=100 \
    dataloader.train.batch_size=16 \
    trainer.save_interval=5000 \
    trainer.run_name="my_experiment"
```

### Resume Training

```bash
python train.py resume_checkpoint=checkpoints/checkpoint_50000.pt
```

### Training Logs

Training logs are automatically uploaded to CometML. View them at:
[https://www.comet.ml/](https://www.comet.ml/)

## Synthesis

### Download Pre-trained Weights

```bash
python download_weights.py
```

This downloads `generator_best.pt` (~55 MB) from Google Drive to `models_weights/`.

### MOS Test Audio

Three ground-truth audio files for MOS evaluation can be downloaded from
[Google Drive](https://drive.google.com/drive/folders/1oClqdJjEGq6nDl9pAhqXivXGD5uJw1RJ):

1. `1.wav`: "Ожидал увидеть тут толпы игроков, жаждущих меня убить, но никому моя голова почему-то не потребовалась."
2. `2.wav`: "Уверяю вас, сэр, большую часть своей жизни я был глубоко убежден, что любой мир состоит из чего угодно, только не из добрых людей."
3. `3.wav`: "Наш самый суровый священник был абсолютно никакой, в хлам, как распоследний питерский сантехник в канун дня праздника Парижской коммуны."

### Synthesize Audio

#### Resynthesize Mode (extract mel from ground-truth audio)

```bash
python synthesize.py \
    input_dir=path/to/audio/folder \
    output_dir=output/synthesized \
    checkpoint=models_weights/generator_best.pt \
    resynthesize=true
```

#### Inference Mode (from acoustic model mel-spectrograms)

```bash
python synthesize.py \
    input_dir=path/to/audio/folder \
    output_dir=output/synthesized \
    checkpoint=models_weights/generator_best.pt \
    resynthesize=false
```

### CustomDirDataset Format

`CustomDirDataset` supports the following directory structure:

```
NameOfTheDirectoryWithUtterances/
├── audio/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── transcriptions/       # optional, for Full TTS pipeline
    ├── file1.txt
    ├── file2.txt
    └── ...
```

- In **resynthesis** mode: audio files from `audio/` are used to extract mel-spectrograms, which are then fed to the vocoder.
- In **Full TTS** mode: transcriptions from `transcriptions/` can be used with an external acoustic model to generate mel-spectrograms, which are then vocoded.

## Model Architecture

### Generator

The generator uses mel-spectrograms as input and upsamples them to raw audio waveforms:

1. **Initial Convolution**: Projects mel-spectrogram to hidden dimension
2. **Upsampling Blocks**: Transposed convolutions + MRF modules
3. **Final Convolution**: Produces 1-channel audio output

**Multi-Receptive Field Fusion (MRF)**: Each MRF module contains multiple residual blocks with different kernel sizes and dilation rates, allowing the model to observe patterns of various lengths.

### Discriminators

#### Multi-Period Discriminator (MPD)
- Consists of 5 sub-discriminators with periods [2, 3, 5, 7, 11]
- Reshapes 1D audio to 2D and applies 2D convolutions
- Captures periodic patterns in speech

#### Multi-Scale Discriminator (MSD)
- Consists of 3 sub-discriminators operating at different scales
- Uses raw audio, 2x pooled, and 4x pooled inputs
- Captures consecutive patterns and long-term dependencies

### Loss Functions

1. **Adversarial Loss (LS-GAN)**: Generator tries to fool discriminators
2. **Mel-Spectrogram Loss**: L1 distance between mel-spectrograms (λ_mel = 45)
3. **Feature Matching Loss**: L1 distance between discriminator features (λ_fm = 2)

## Configuration

### Mel-Spectrogram Parameters

| Parameter | Value |
|-----------|-------|
| Sample Rate | 22050 Hz |
| FFT Size | 1024 |
| Window Size | 1024 |
| Hop Size | 256 |
| Mel Bands | 80 |
| F_min | 0 Hz |
| F_max | 8000 Hz |

### Model Configuration (V1)

| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 512 |
| Parameters | ~14M |
| Upsample Rates | [8, 8, 2, 2] |
| Upsample Kernels | [16, 16, 4, 4] |
| ResBlock Kernels | [3, 7, 11] |
| ResBlock Dilations | [[1,3,5], [1,3,5], [1,3,5]] |

## Demo

A Google Colab compatible demo notebook is provided in `demo.ipynb`.

To run the demo:
1. Open `demo.ipynb` in Google Colab
2. Run all cells
3. The notebook will:
   - Clone the repository
   - Install dependencies
   - Download pre-trained weights
   - Demonstrate audio synthesis

## Report

### Model Description

HiFi-GAN is a GAN-based neural vocoder that synthesizes high-fidelity speech audio from mel-spectrograms. Key innovations:

1. **Multi-Period Discriminator**: Captures periodic patterns by processing audio at different periods
2. **Multi-Scale Discriminator**: Evaluates audio at multiple scales for long-term dependencies
3. **Multi-Receptive Field Fusion**: Combines outputs from multiple residual blocks with diverse receptive fields

### Training Details

- **Dataset**: RUSLAN (Russian language, single speaker)
- **Optimizer**: AdamW (β1=0.8, β2=0.99, weight_decay=0.01)
- **Learning Rate**: 2×10⁻⁴ with exponential decay (γ=0.999)
- **Batch Size**: 16
- **Segment Size**: 8192 samples (~0.37 seconds at 22050 Hz)

### Results

Training logs are available on CometML: [HiFi-GAN Russian TTS](https://www.comet.com/krugd/hifigan-russian-tts/444cc4e3c69c439cbd2f2186e108e03c?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step)

### Challenges

1. **Memory constraints**: Training with full audio requires significant GPU memory
2. **Training stability**: Balancing generator and discriminator training
3. **Dataset preprocessing**: Ensuring consistent mel-spectrogram extraction

## References

```bibtex
@inproceedings{kong2020hifi,
  title={HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis},
  author={Kong, Jungil and Kim, Jaehyeon and Bae, Jaekyoung},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## License

This project is for educational purposes as part of a Deep Learning for Audio course homework assignment.
