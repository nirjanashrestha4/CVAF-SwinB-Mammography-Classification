# CVAF-SwinB: Cross-View Asymmetric Fusion Swin Transformer for Breast Cancer Classification

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF.svg)](https://kaggle.com)

> **Thesis:** Enhance Vision Transformer Based Breast Cancer Mammography Classification  
> **Author:** Nirjana Shrestha  
> **Institution:** Melbourne Institute of Technology (MIT)  
> **Year:** 2026

---

## Overview

This repository contains the official implementation of **CVAF-SwinB**, a novel
Vision Transformer architecture for binary breast cancer classification using
four-view screening mammography (L-CC, L-MLO, R-CC, R-MLO).

Unlike existing single-view or naive multi-view approaches, CVAF-SwinB explicitly
models the **clinical reading hierarchy** used by radiologists:

- **Asymmetric Lateral Fusion** — directional cross-attention between CC and MLO
  views within the same breast, mirroring how radiologists compare depth and
  structural context
- **Bilateral Contralateral Fusion** — cross-attention between left and right
  breast representations to detect bilateral asymmetry, a key clinical indicator
  of malignancy
- **Gated Global Aggregation** — a learned gating mechanism that assigns
  patient-specific importance weights to each view for final prediction

---

## Model Architecture

The CVAF-SwinB pipeline consists of four sequential stages:

