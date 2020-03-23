# SEAN: Image Synthesis with Semantic Region-Adaptive Normalization (CVPR 2020 Oral)

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.2.0](https://img.shields.io/badge/pytorch-1.2.0-green.svg?style=plastic)
![pyqt5 5.13.0](https://img.shields.io/badge/pyqt5-5.13.0-green.svg?style=plastic)

![image](./docs/assets/Teaser.png)
**Figure:** *Face image editing controlled via style images and segmentation masks with SEAN*

We propose semantic region-adaptive normalization (SEAN), a simple but effective building block for Generative Adversarial Networks conditioned on segmentation masks that describe the semantic regions in the desired output image. Using SEAN normalization, we can build a network architecture that can control the style of each semantic region individually, e.g., we can specify one style reference image per region. SEAN is better suited to encode, transfer, and synthesize style than the best previous method in terms of reconstruction quality, variability, and visual quality. We evaluate SEAN on multiple datasets and report better quantitative metrics (e.g. FID, PSNR) than the current state of the art. SEAN also pushes the frontier of interactive image editing. We can interactively edit images by changing segmentation masks or the style for any given region. We can also interpolate styles from two reference images per region.

> **SEAN: Image Synthesis with Semantic Region-Adaptive Normalization** <br>
> Peihao Zhu, Rameen Abdal, Yipeng Qin, Peter Wonka <br>
> *Computer Vision and Pattern Recognition **CVPR 2020, Oral***


[[Paper](https://arxiv.org/pdf/1911.12861.pdf)]
[[Project Page](https://zpdesu.github.io/SEAN/)]
[[Demo](https://youtu.be/0Vbj9xFgoUw)]


## Installation
### Code will be uploaded within this week

## Dataset Preparation

## Pretrained Models

## Training & Testing Models

## UI Introduction
Yes, We will release the UI. We hope to make the image editing tasks easier for everyone.

## License

All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**) The code is released for academic research use only.

## Citation
If you use this code for your research, please cite our papers.
```
@misc{zhu2019sean,
    title={SEAN: Image Synthesis with Semantic Region-Adaptive Normalization},
    author={Peihao Zhu and Rameen Abdal and Yipeng Qin and Peter Wonka},
    year={2019},
    eprint={1911.12861},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgments
We thank Wamiq Reyaz Para for helpful comments. This code borrows heavily from SPADE. We thank Taesung Park for sharing his codes. This work was supported by the KAUST Office of Sponsored Research (OSR) under AwardNo. OSR-CRG2018-3730.
