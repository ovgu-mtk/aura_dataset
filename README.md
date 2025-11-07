# Aura Dataset - A BikePerception Dataset 

**A vision dataset for semantic segmentation and object detection recorded from a bike's perspective.**

This repository provides sample data and code for working with a custom dataset focused on urban mobility 
and perception tasks from the viewpoint of a bicycle. It includes data for **semantic segmentation** and 
**bounding box detection**, a TensorFlow-based training pipeline, and utilities for visualization.

> 📧 **Note**: Only sample data is included here. To request access to the full dataset for 
> academic or research purposes, please contact:  
> [stefan.sass@ovgu.de](mailto:stefan.sass@ovgu.de)  
> [markus.hoefer@ovgu.de](mailto:markus.hoefer@ovgu.de)

---

## 📸 Sensor Setup

The dataset was recorded from a mobile sensor platform mounted on a cargo bike, 
capturing urban, park, and pedestrian areas.
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
│   └── semantic_viewer.py  # Visualize semantic segmentation 
│   └── classes.py          # Class descriptions for bb/semantic

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

### Bounding Boxes:

- YOLOv8 Models with backbones:
  - YOLOv8-XS
  - YOLOv8-L

**Table: COCO-style evaluation metrics for YOLOv8-XS and YOLOv8-L**

| **Metric**                | **YOLOv8-XS [%]** | **YOLOv8-L [%]** |
|--------------------------|------------------|------------------|
| mAP@[0.5:0.95]           | 14.4             | 17.2             |
| mAP@0.5                  | 20.2             | 23.2             |
| mAP@0.75                 | 16.8             | 20.1             |
| mAP (Small objects)      | 3.1              | 10.4             |
| mAP (Medium objects)     | 16.7             | 19.7             |
| mAP (Large objects)      | 18.0             | 23.3             |
| Recall@1                 | 11.7             | 13.4             |
| Recall@10                | 16.8             | 20.1             |
| Recall@100               | 16.8             | 20.2             |
| Recall (Small objects)   | 3.9              | 11.2             |
| Recall (Medium objects)  | 18.7             | 21.6             |
| Recall (Large objects)   | 21.1             | 27.0             |


![semantic_results.png](doc/bb_results.png)
Qualitative comparison of object detection results. From left to right: 
Original image with ground truth annotations, YOLOv8-XS prediction, YOLOv8-L prediction.


### Semantic:


| **Class Granularity**                 | **mPA [%]** | **mIoU [%]** |
|--------------------------------------|-------------|--------------|
| 25 classes (all individual labels)   | 67.9        | 48.07        |
| 19 classes (partially merged)        | 69.36       | 57.13        |
| 13 classes (strongly merged)         | 76.7        | 67.2         |


![semantic_results.png](doc/semantic_results.png)
Qualitative comparison of semantic segmentation results. Each column shows one example. 
The first row contains the original input images. The second row shows the ground truth annotations. 
Rows three to five show the predictions from models trained with 13, 19, and all 25 classes, respectively. 


## 📩 Requesting Full Dataset

This repository contains only sample data. For access to the complete dataset (size ~5.9 GB), please contact:

    stefan.sass@ovgu.de

    markus.hoefer@ovgu.de

Please include your institutional affiliation and intended use (academic or research only).


## 📄 License

This dataset is provided for academic and research use only. 
Redistribution or commercial usage is not permitted without explicit permission.


## 🔗 Citation

Please use the following citation when referencing https://rdcu.be/eOJtd:

```
@article{aura2025,
  title = {AuRa dataset: A vision dataset from a bike’s perspective for autonomous robots in urban environments},
  ISSN = {2366-598X},
  url = {http://dx.doi.org/10.1007/s41315-025-00502-x},
  DOI = {10.1007/s41315-025-00502-x},
  journal = {International Journal of Intelligent Robotics and Applications},
  publisher = {Springer Science and Business Media LLC},
  author = {Sass,  Stefan and H\"{o}fer,  Markus and Weißflog,  Julia and Schmidt,  Stephan and Scholz,  Andreas},
  year = {2025},
  month = nov 
}
```
