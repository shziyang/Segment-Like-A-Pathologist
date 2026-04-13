# Segment-Like-A-Pathologist

This repository contains the official PyTorch implementation of the following paper:

#### Segment Like A Pathologist: Learning reliable diagnostic thinking and experience for Precise Plant Pests and Diseases Segmentation in Complex Environment

Xuan Xiong, Ziyang Shi, Hao Zhou, Wei Shi, Fulin Su, Lin Li , Shaofeng Peng, Ruifeng Liu, Fangying Wan
School of Electronic Information and Physics, Central South University of Forestry and Technology  

## Abstract
Deep learning-driven disease segmentation has emerged as a pivotal technology for crop health monitoring and intelligent plant protection, underpinning the sustainable development of the Camellia oleifera industry. However, prevailing methods encounter three core challenges in complex agricultural environments: (1) excessive model parameters and computational complexity, which hinder deployment on edge devices; (2) inadequate integration of global context, limiting the capture of long-range spatial dependencies; and (3) limited robustness in complex field settings, rendering models susceptible to illumination variations and background interference. Furthermore, prevailing approaches often operate as "black boxes" that rely solely on data fitting without explicitly incorporating the diagnostic priors and logic of plant pathologists, thereby limiting interpretability and decision-making trustworthiness. To address these limitations, we propose the SLAP (Segment Like A Pathologist) model. This model enhances accuracy and interpretability by mimicking the progressive reasoning workflow of plant pathologists—spanning disease classification, severity quantification, and control strategy generation. First, we employ Vision Mamba as the backbone to balance efficiency and performance via its linear-complexity global modeling capabilities. Second, we introduce a Context Clustering Vision Mamba (CCViM) module, which achieves progressive multi-scale feature extraction by dynamically aggregating local disease features and integrating global context through bidirectional scanning. Subsequently, a Causal Graph Reasoning Module (CGRM) is constructed to explicitly model the topological distinctions between lesions and healthy tissues, enhancing discriminative capability in complex backgrounds. Experimental results demonstrate that SLAP achieves superior performance on both self-built and public datasets. Specifically, it outperforms the baseline VMUNet with improvements of 5.19% and 4.23% in mIoU and Dice coefficient, respectively. Furthermore, SLAP surpasses TransUNet by 3.99% in mIoU, while significantly minimizing model complexity, achieving reductions of 68.85 M in parameters and 20.07 GFLOPs in computational cost. To validate practical utility, we integrated the open-source multimodal large model, Qwen3-VL, to develop the Camellia oleifera Pests and Disease Diagnosis System (CDDS). This system establishes an intelligent closed-loop framework for "feature recognition, pathological analysis, and decision support," capable of not only pixel-level segmentation and area quantification but also generating pathology-informed control recommendations, demonstrating promising potential for field application.


## The overall architecture
![framework](image/fig1.png)

## Visual results on Camellia oleifera pest and disease dataset
![framework](image/fig2.png)

## Visual results on AL9EE apple disease dataset
![framework](image/fig3.png)
