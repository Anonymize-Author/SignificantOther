# SignificantOther


## News
* 2024.10.28 The model weights have been open at [🤗SignificantOther-7B model](https://huggingface.co/spaces/Ice-lemon/SignificantOtherDemo/tree/main) . The model was trained on both dfew and ferv39k.

* 2024.10.30 The Dataset will be open at [🤗SO Dataset-40k](https://huggingface.co/datasets/Ice-lemon/SOdataset/tree/main). The dataset will be made publicly available after the paper is accepted.

* 2024.11.4 The demo is available at [🤗SO Demo](https://huggingface.co/spaces/Ice-lemon/SignificantOtherDemo), but due to resource constraints we cannot keep it open for a long time. The demo can be run on any graphics card with more than 24G of video memory.


## Key Enhancements

* We propose a multi-stage training paradigm to gradually enhance the model's emotion understanding capabilities. Stage one handles atomic tasks like video understanding, subject identification, and expression analysis, providing foundational skills; stage two introduces chain-of-thought reasoning, connecting atomic tasks for integrated emotional inference; stage three completes large-scale fine-tuning with DFEW and FERV39k, ensuring cross-modal, cross-temporal reliability.
* To support this three-stage training, we developed a reliable dataset, SO dataset, which decompose the emotion understanding task into multiple atomic and chain-of-thought tasks, with annotations cross-validated by two advanced vision-language models and human reviewers. This approach greatly reduces annotation costs while enhancing data quality.
* By combining this structured training and reliable dataset, Significant Other achieves SOTA performance(5.54\% to 6.37\%) as the first unified foundation model for emotion understanding, seamlessly integrating time sequences, scenarios, and multimodal data for comprehensive emotion recognition. All weights, codes and datasets will be open source.

## Performance

### Results of 7 Classes on DFEW and FERV39k

**Table 1**: Results of 7 classes on DFEW and FERV39k. Bold represents the optimal result and underscore represents the suboptimal.

| Method              | Publication                | Happy   | Sad     | Neutral | Angry   | Surprise | Disgust | Fear   | UAR   | WAR   |
|---------------------|----------------------------|---------|---------|---------|---------|----------|---------|--------|-------|-------|
| VGG13+LSTM          | /                          | 76.89   | 37.65   | 58.04   | 60.7    | 43.70    | 0.00    | 19.73  | 42.39 | 53.70 |
| C3D [Tran et al., 2015] | CVPR’15                 | 75.17   | 39.49   | 55.11   | 62.49   | 45.00    | 1.38    | 20.51  | 42.74 | 53.54 |
| ResNet18+LSTM       | /                          | 83.56   | 61.56   | 68.27   | 65.29   | 51.26    | 0.00    | 29.34  | 51.32 | 63.85 |
| ResNet18+GRU        | /                          | 82.87   | 63.83   | 65.06   | 68.51   | 52.00    | 0.86    | 30.14  | 51.68 | 64.02 |
| I3D-RGB [Carreira et al., 2017] | CVPR’17         | 78.61   | 44.19   | 56.69   | 55.87   | 45.88    | 2.07    | 20.51  | 43.40 | 54.27 |
| P3D [Qiu et al., 2017] | ICCV’17                 | 74.85   | 43.40   | 54.18   | 60.42   | 50.99    | 0.69    | 23.28  | 43.97 | 54.47 |
| R(2+1)D18 [Hara et al., 2018] | CVPR’18          | 79.67   | 39.07   | 57.66   | 50.39   | 48.26    | 3.45    | 21.06  | 42.79 | 53.22 |
| 3D R18+Center Loss  | /                          | 78.49   | 44.30   | 54.89   | 58.40   | 52.35    | 0.69    | 25.28  | 44.91 | 55.48 |
| 3D ResNet18         | CVPR’18                    | 76.32   | 50.21   | 64.18   | 62.85   | 47.52    | 0.00    | 24.56  | 46.52 | 58.27 |
| EC-STFL [Jiang et al., 2020] | MM’20             | 79.18   | 49.05   | 57.85   | 60.98   | 46.15    | 2.76    | 21.51  | 45.35 | 56.51 |
| Former-DFER [Zhao et al., 2021] | MM’21          | 84.05   | 62.57   | 67.52   | 70.03   | 56.43    | 3.45    | 31.78  | 53.69 | 65.70 |
| NR-DFERNet [Li et al., 2022] | arXiv’22          | 88.47   | 64.84   | 70.03   | 75.09   | 61.60    | 0.00    | 19.43  | 54.21 | 68.19 |
| GCA+IAL [Li et al., 2023] | C&C23                | 87.95   | 67.21   | 70.10   | 76.06   | 62.22    | 0.00    | 26.44  | 55.71 | 69.24 |
| SW-FSCL [Yan et al., 2023] | AAAI’23             | 88.35   | 68.52   | 70.98   | **78.17** | 64.25  | 1.42    | 28.66  | 57.25 | 70.81 |
| M3DFEL [Wang et al., 2023] | CVPR’23             | 89.59   | 68.38   | 67.88   | 74.24   | 59.69    | 0.00    | 31.64  | 56.10 | 69.25 |
| LSGTNet [Wang et al., 2024] | Appl Soft Comput’24 | **90.67** | 71.70 | 70.48   | 76.71   | _65.01_  | 14.48   | 40.24  | 61.33 | 72.34 |
| OUS [Mai et al., 2024] | /                      | **94.40** | _83.23_ | _71.03_ | 77.33   | 60.98    | **31.01** | 34.12  | 64.33 | 74.02 |
| S2D [Chen et al., 2024] | IEEE T Affect Comput’24 | _93.95_  | 78.35 | 70.25   | _78.00_ | 61.88    | _25.52_ | _50.22_ | _65.45_ | _74.81_ |
| **SO (ours)**       | /                          | 85.48   | **86.02** | **87.83** | 76.55   | **67.69** | 3.75    | **75.14** | **68.88** | **80.39** |

---


### Results on DFEW and FERV39k

**Table 2**: Results on DFEW and FERV39k. Bold represents the optimal result and underscore represents the suboptimal.

| Method             | DFEW UAR | DFEW WAR | FERV39k UAR | FERV39k WAR |
|--------------------|----------|----------|-------------|-------------|
| C3D [Tran et al., 2015]  | 42.74    | 53.54    | 22.68       | 31.69       |
| P3D [Qiu et al., 2017]   | 43.97    | 54.47    | 23.20       | 33.39       |
| I3D-RGB [Carreira et al., 2017] | 43.40 | 54.27 | 30.17       | 38.78       |
| 3D ResNet18        | 46.52    | 58.27    | 26.67       | 37.57       |
| R(2+1)D18          | 42.79    | 53.22    | 31.55       | 41.28       |
| ResNet18-LSTM      | 51.32    | 63.85    | 30.92       | 42.95       |
| ResNet18-ViT       | 55.76    | 67.56    | 38.35       | 48.43       |
| EC-STFL [Jiang et al., 2020] | 45.35 | 56.51 | -           | -           |
| Former-DFER        | 53.69    | 65.70    | 37.20       | 46.85       |
| NR-DFERNet [Li et al., 2022] | 54.21 | 68.19 | 33.99       | 45.97       |
| DPCNet [Wang et al., 2022]  | 57.11    | 66.32    | -           | -           |
| EST                | 53.94    | 65.85    | -           | -           |
| LOGO-Former [Ma et al., 2023] | 54.21 | 66.98 | 38.22       | 48.13       |
| IAL [Li et al., 2023]      | 55.71    | 69.24    | 35.82       | 48.54       |
| CLIPER [Li et al., 2023]   | 57.56    | 70.84    | 41.23       | 51.34       |
| M3DFEL [Wang et al., 2023] | 56.10    | 69.25    | 35.94       | 47.67       |
| AEN                | 56.66    | 69.37    | 38.18       | 47.88       |
| DFER-CLIP [Zhao et al., 2023] | 59.61 | 71.25 | 41.27       | 51.65       |
| EmoCLIP [Foteinopoulou et al., 2023] | 58.04 | 62.12 | 31.41 | 36.18       |
| LSGTNet [Wang et al., 2024] | 61.33 | 72.34 | 41.30       | 51.31       |
| MMA-DFER [Chumachenko et al., 2024] | _67.01_ | _77.51_ | -           | -           |
| UniLearn [Chen et al., 2024] | 66.80 | 76.68 | _43.41_    | _53.65_    |
| OUS [Mai et al., 2024]     | 64.33    | 74.02    | 42.43       | 53.30       |
| **SO (ours)**      | **68.88** | **80.39** | **53.69**  | **58.84**  |
