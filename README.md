# Microplastic Detection and Acoustic Wave Water Filtration System

This project presents an integrated system for detecting microplastic particles in water samples and initiating an acoustic wave-based water filtration process.

The system combines a YOLOv8-based object detection model with an acoustic wave water filtration system. When the model detects microplastic particles in a water sample, it sends a response to the filtration system to initiate the water filtration process.

## Dataset

The dataset consists of approximately 1,200 images of water samples used to develop and evaluate the microplastic detection model.

## Model

The project uses the YOLOv8 object detection model to detect microplastic particles in water sample images.

The model analyzes the input image and determines whether microplastic particles are present. When microplastic particles are detected, the model provides a response that is used to trigger the connected filtration system.

## System Integration

The detection model is integrated with an acoustic wave-based water filtration system.

The integration follows a simple workflow:

1. A water sample is analyzed by the YOLOv8 model.
2. The model detects whether microplastic particles are present.
3. If microplastic particles are detected, the model sends a response to the filtration system.
4. The acoustic wave filtration system is activated to filter the contaminated water.
5. The filtered water can then be evaluated for further verification.

## Development Tools

- Python   - YOLOv8   - Deep Learning   - Computer Vision   - Acoustic Wave Technology   - Roboflow
