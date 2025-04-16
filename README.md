# Aura Dataset - A BikePerception Dataset 

**A vision dataset for semantic segmentation and object detection recorded from a bike's perspective.**

This repository provides sample data and code for working with a custom dataset focused on urban mobility and perception tasks from the viewpoint of a bicycle. It includes data for **semantic segmentation** and **bounding box detection**, a TensorFlow-based training pipeline, and utilities for visualization.

> 📧 **Note**: Only sample data is included here. To request access to the full dataset for academic or research purposes, please contact:  
> [stefan.sass@ovgu.de](mailto:stefan.sass@ovgu.de)  
> [markus.hoefer@ovgu.de](mailto:markus.hoefer@ovgu.de)

---

## 📸 Sensor Setup

The dataset was recorded from a mobile sensor platform mounted on a cargo bike, capturing urban, park, and pedestrian areas.
It reflects the unique perspective and challenges of micromobility in diverse environments.

![sensor_platform.jpg](doc/aura_bike.jpg)


## 🗂️ Folder Structure

```
├── dataset
│   ├── bb                  # Bounding box data
│   │   ├── train_val/
│   │   └── test/
│   └── semantic            # Semantic segmentation data
│       ├── train_val/
│       └── test/
│
├── training
│   ├── bb/                 # Training scripts and saved models for bounding box detection
│   └── semantic/           # Training scripts and saved models for semantic segmentation
│
├── utils
│   ├── bb_viewer.py        # Draw bounding boxes on images 
│   └── semantic_viewer.py
│   └── classes.py 

```

## 🧠 Tasks

### Semantic Segmentation

- Pixel-wise classification of urban scenes from the cyclist's point of view.

- Includes classes like pedestrians, cyclists, vehicles, infrastructure, etc.

- Ground truth provided as annotated label images.

![semantic_examples.png](doc/semantic_examples.png)



### Bounding Box Detection

- Object detection in 2D using bounding boxes.

- Useful for detecting dynamic agents and static objects in shared mobility spaces.

![bb_examples.png](doc/bb_examples.png)

## 🔬 Baseline Models

Baseline models are provided using TensorFlow. Each task comes with:

- Custom dataloaders

- Model architecture

- Training and testing scripts

- Pretrained models 

You can find them under:

- training/semantic/ for semantic segmentation

- training/bb/ for bounding box detection


## 📩 Requesting Full Dataset

This repository contains only sample data. For access to the complete dataset, please contact:

    stefan.sass@ovgu.de

    markus.hoefer@ovgu.de

Please include your institutional affiliation and intended use (academic or research only).


## 📄 License

This dataset is provided for academic and research use only. 
Redistribution or commercial usage is not permitted without explicit permission.


## 🔗 Citation

A citation entry will be added here once the corresponding paper or technical report is published.