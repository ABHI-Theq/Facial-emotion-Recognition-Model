# Facial Sentiment Analysis CNN

This repository contains a Convolutional Neural Network (CNN) implementation for facial emotion recognition, capable of classifying facial expressions into multiple emotional categories.

## Project Overview

This deep learning model can classify facial expressions into six emotional categories:
- Ahegao
- Angry
- Happy
- Neutral
- Sad
- Surprise

The implementation uses TensorFlow/Keras with a custom CNN architecture featuring multiple convolutional layers with batch normalization for improved training stability and performance.

## Dataset

The model is trained on the [Emotion Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset/data) from Kaggle, which contains facial expression images categorized into different emotions. The dataset has the following distribution:

```
Full dataset class counts: {'Ahegao': 1205, 'Angry': 1312, 'Happy': 3739, 'Neutral': 4025, 'Sad': 3933, 'Surprise': 1234}
```

The data is split into training (80%), validation (10%), and test (10%) sets with stratified sampling to maintain the class distribution.

## Model Architecture

The model uses a deep CNN architecture with the following key components:

- Multiple convolutional blocks, each consisting of:
  - 2D Convolutional layer
  - Batch Normalization
  - ReLU Activation
  - Max Pooling

- The network progressively increases filter depth (32 → 64 → 128 → 256 → 512 → 1024 → 2048)
- Global Average Pooling to reduce spatial dimensions
- Multiple Dense layers with dropout for regularization
- Final softmax layer for classification

The complete architecture contains approximately 27.9 million parameters.

## Implementation Details

### Data Preprocessing

- Images are resized to 224×224 pixels
- Data augmentation techniques applied:
  - Horizontal flipping
  - Rotation (±15 degrees)
  - Zoom (up to 20%)
  - Width/height shifting (up to 10%)

### Training

- Weighted loss function to handle class imbalance
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Model checkpoint to save the best performing model
- Batch size of 64

### Performance

The model achieves approximately 80.78% accuracy on the test set after training. The training process shows good convergence with the validation loss decreasing from ~1.58 to ~0.46.

## Usage

### Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Running the Model

1. Download the [Emotion Recognition Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset/data)
2. Update the data directory path in the notebook
3. Run the notebook cells sequentially

## Results

The model was trained for approximately 32 epochs before early stopping. The best model achieved:
- Training Accuracy: ~83.14%
- Validation Accuracy: ~82.52%
- Test Accuracy: ~80.78%

## Future Improvements

- Experiment with pretrained models like VGG, ResNet, or EfficientNet
- Implement cross-validation for more robust evaluation
- Try different augmentation techniques specific to facial expressions
- Explore attention mechanisms to focus on important facial regions

## License

[Include your license information here]

## Acknowledgements

- Dataset from [Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset/data)
- Inspired by various facial emotion recognition research papers and implementations
