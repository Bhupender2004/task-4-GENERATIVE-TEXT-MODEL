# Generative Text Model

## Overview
This project implements a Generative Text Model using Long Short-Term Memory (LSTM) neural networks. The model is designed to generate coherent paragraphs based on input prompts. It utilizes TensorFlow and Keras for building and training the neural network.

## Features
- Implements an LSTM-based neural network for text generation.
- Supports dynamic text generation based on user-defined prompts.
- Preprocesses textual data using tokenization and padding techniques.
- Outputs generated text by predicting word sequences iteratively.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy

### Install Dependencies
```bash
pip install tensorflow numpy
```

## File Structure
- `generative_text_model.py`: Contains the complete implementation of the generative text model.
- `README.md`: Provides an overview of the project, installation instructions, and usage guidelines.

## Usage
1. Run the script:
```bash
python generative_text_model.py
```
2. Enter a seed text and specify the number of words to generate.
3. The model generates a coherent paragraph based on the input.

### Example:
Input:
```
Seed Text: Artificial Intelligence
Next Words: 10
```
Output:
```
Artificial Intelligence is a field that focuses on creating intelligent systems and algorithms for various applications.
```

## Model Training
- The model is trained on a predefined dataset containing text about AI, Machine Learning, and NLP.
- It uses categorical cross-entropy as the loss function and Adam optimizer for training.

## Customization
- Modify the `corpus` variable in the code to include your custom dataset.
- Adjust the model parameters (e.g., LSTM layers, dropout rate) for performance tuning.



