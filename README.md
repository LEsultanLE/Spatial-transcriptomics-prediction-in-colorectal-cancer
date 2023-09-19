# Spatial transcriptomics prediction in colorectal cancer
This project's aim was to predict spatial gene expression in colorectal cancer (CRC) from H&E whole-slide images. It consists of three parts: 
1. Developing a CNN-based CRC tissue classifier based on publicly available datasets of CRC tissue classes (NCT-CRC-HE-100K and CRC-VAL-HE-7K [^1]) for a same-domain transfer learning approach.
2. Pre-processing in-situ sequencing (CARTANA) spatial transcriptomics (ST) data for further use.
3. Developing single-gene and multi-gene CNN-based models to predict spatial gene expression patterns from selected genes.

[^1]: 10.5281/zenodo.1214456

## CRC tissue classifier
In this part, three different deep learning models using EfficientNet [^2], GoogleNet [^3] and Resnet50 [^4] as backbone architectures were trained on 85% of the NCT-CRC-HE-100K dataset and subsequently tested on the hold-out data as well as the external testing set CRC-VAL-HE-7K. Further optimization or hyperparameter tunings were not performed as these models were purely developed for transfer learning and retrained on the spatial transcriptomics data later.

[^2]: arXiv:1905.11946
[^3]: 10.1109/CVPR.2015.7298594
[^4]: 10.1109/CVPR.2016.90

## Pre-processing ST dataset
In the second part, the ST dataset produced by in-situ sequencing using CARTANA probes for colorectal cancer was pre-preprocessed so that it could be used for CNN-based deep learning models. Here, gene-location matrices were transformed into a pseudo-spot dataset by summing up all gene counts per pre-defined tissue area that matched the size of the training image tiles from NCT-CRC-HE-100K. Background as well as low gene counts pseudo-spots were filtered out. Total gene counts per CRC sample were adjusted using the trimmed mean of M values method [^5]. Batch effect correction was achieved with the ComBat algorithm [^6]. 

[^5]: 10.1093/biostatistics/kxj037
[^6]: 10.1186/gb-2010-11-3-r25

## Prediction of spatial gene expression patterns
Lastly, transformed count values of six single genes (RAMP1, RUBCNL, PIGR, TNS1 = consensus molecular subtype genes; COL1A1 and COL1A2 = stromal marker genes) were predicted. In the first approach, CNN models were trained using different pre-trained (CRC tissue classifier) parameters as well as different batch sizes and learning rates. During training MSE loss and Pearson's r correlation coefficient (PCC) were monitored and the model parameters with the best performing PCC on the validation dataset were saved for further testing. In the second approach, models were trained for the simultaneous predictions of all consensus molecular subtype genes or all six genes.
