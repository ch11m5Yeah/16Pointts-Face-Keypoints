# 16-Point Face Detection

## Introduction

This project implements a **16-point facial landmark detection model**.  
The model is designed for embedded deployment on **RA8D1**, achieving around **22 ms per frame**.

The architecture features a compact backbone and a **YOLO-style region-wise regression head** for keypoint prediction.

## Features

- Input size: **96 × 96 pixels**  
- Output channels: **48**  
  - Every 3 channels correspond to one keypoint:  
    1. Confidence score  
    2. X offset relative to the grid cell center  
    3. Y offset relative to the grid cell center  
- Lightweight and efficient  
- Embedded-friendly, optimized for **RA8D1**  
- Inference speed: ~22 ms per frame  
- Supports quantization  

## Dataset

- Uses **LabelMe** annotation format  
- **Requirement:** Each annotation JSON file **must include the original image**  
  - Either as a **base64-encoded image** embedded in the JSON, or as a **path pointing to the image file**  
- To train on your own dataset, modify the files in the `dataset` folder  

## Training

- Hyperparameters can be adjusted in `train.py`  
- `spsf_quant` controls quantization; adjust as needed  
- Training scripts require JSON files to contain both labels and images  

## Deployment

- Provides **C files** generated via **Renesas eAI tools**  
- Inference accelerated using **ARM CMSIS-NN**  
- Decoding predictions can be referenced in the provided Python code or C implementation  

## Limitations

- Accuracy decreases on **blurred faces**  

## Notes

- This is a learning-stage project, meant for experimentation and embedded deployment practice.
