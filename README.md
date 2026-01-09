# PDAE-ESWA

## Patch distance based auto-encoder for industrial anomaly detection

The official code of the paper " *Patch distance based auto-encoder for industrial anomaly detection*", which is accepted by "
Expert Systems with Applications". [PDAE](https://doi.org/10.1016/j.eswa.2025.126537)

## Abstract

Industrial anomaly detection (IAD) aims to classify images and locate defective areas of anomalous samples. To perform IAD, auto-encoder is widely adopted to train on anomaly-free images and infer by calculating the distance between the input and the reconstructed output. However, in practice, auto-encoder cannot restore the anomalous region well, as it loses small spatial information, leading to the unpromising performance of detection. To overcome the above limitation, a novel approach named Patch Distance Based Auto-Encoder For Industrial Anomaly Detection (PDAE) is proposed in this study. To increase the receptive field size and robustness to small spatial deviations, patch-level features are aggregated to the reconstruction process, which can help to better differentiate and locate the anomalous areas. Furthermore, to better classify the samples, cross-patch score and cross-dimension score are incorporated in the inference stage. Moreover, to reduce the bias towards the pre-trained dataset, PDAE incorporates multi-level features extracted from convolutional neural networks and class-attention in image transformers. We conducted experiments on the MVTec-AD benchmark dataset and achieved state-of-the-art performance, with 99.7 % and 98.49% AUROC scores in anomaly detection and localization, respectively. In conclusion, extensive experimental results show that our method outperforms some state-of-the-art baseline methods on the used metrics and addresses the limitations effectively.


## Getting Started

- `python main.py`

## Citation

If you find this code useful, don't forget to star the repo and cite the paper:

```
@article{MA2025126537,
title = {Patch distance based auto-encoder for industrial anomaly detection},
journal = {Expert Systems with Applications},
volume = {270},
pages = {126537},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.126537},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425001599},
author = {Zeqi Ma and Jiaxing Li and Wai Keung Wong},
keywords = {Patch, Anomaly detection, Reduce bias, Convolutional neural network, Transformer}
}
```



## Acknowledgements

We thank the great works [Cutpaste](https://github.com/Lieberk/Cutpaste), [CDO](https://github.com/caoyunkang/CDO), [REB](https://github.com/ShuaiLYU/REB) for providing assistance for our research.