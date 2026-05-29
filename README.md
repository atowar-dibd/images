# ChandraOCR sample output

This repository contains images and their corresponding text extracted using **ChandraOCR**.

## Contents

- `images/` — Source images processed by the OCR system
- `outputs/` — Extracted text outputs corresponding to each image
- 'infere_batch.py' - Batch inference scripts
- 'inference.py' - Performing inference on single sample image
  
## Description

The dataset was generated using ChandraOCR, an OCR pipeline designed to extract structured text from images. Each image in the dataset has a corresponding text file containing the recognized content.

This repository is intended for research and development purposes, including OCR benchmarking, dataset creation, and downstream NLP tasks.

## Notes

- The mapping between images and extracted text is consistent via filename alignment.
- Text outputs may contain OCR errors depending on image quality and complexity
