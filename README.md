# Handwritten Digit Recognition

[![Watch the video](https://img.youtube.com/vi/your-video-id/maxresdefault.jpg)](https://youtu.be/your-video-id)

## Try it Now!
Check out the live project on Streamlit: [Handwritten Digit Recognition](https://digitrecognitionml.streamlit.app/)

## Overview
This repository contains a Streamlit application for recognizing handwritten digits using a Convolutional Neural Network (CNN) model trained on the MNIST dataset. The model achieves an impressive accuracy of 99.14%. The application allows users to draw a digit on a canvas and instantly get a prediction along with the prediction probabilities.

### Project Highlights
- **Data Preprocessing**: Involves loading and normalizing the MNIST dataset. The images are reshaped to include a channel dimension and normalized to have pixel values between 0 and 1. Labels are converted to one-hot encoded vectors.
- **Model Building**: A CNN is constructed using Keras with layers for convolution, pooling, flattening, and dense connections. This architecture is effective for recognizing patterns in image data.
- **Model Training and Evaluation**: The model is trained on the MNIST dataset and achieves an accuracy of 99.14% on the test dataset. Evaluation includes measuring accuracy and loss on both training and test datasets.
- **Interactive Web Application**: Streamlit is used to create an interactive web interface where users can draw digits and see predictions in real-time. The app visualizes the input image, processed image, and prediction probabilities.

## File Structure
```
.
├── .streamlit/
├── model/
│   └── mnist_model.h5
├── prediction/
├── .gitignore
├── App.py
├── gambar.png
├── model.ipynb
└── requirements.txt
```

### File Descriptions
- **`.streamlit/`**: Contains configuration files for Streamlit.
- **`model/`**: Contains the trained model (`mnist_model.h5`).
- **`prediction/`**: Directory for storing prediction results.
- **`.gitignore`**: Specifies files and directories to be ignored by git.
- **`App.py`**: Main Streamlit application file.
- **`gambar.png`**: Image used in the sidebar of the application.
- **`model.ipynb`**: Jupyter Notebook used for training the CNN model.
- **`requirements.txt`**: Lists all the dependencies required to run the application.

## How to Run the Application

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application**:
   ```bash
   streamlit run App.py
   ```

4. **Interact with the Application**:
   - Draw a digit on the canvas.
   - Click the "Predict" button to see the predicted digit and the prediction probabilities.

## Project Workflow

1. **Data Loading and Preprocessing**:
   - **Loading Data**: Load the MNIST dataset using Keras.
   - **Expanding Dimensions**: Reshape images to include a channel dimension.
   - **Normalizing Data**: Normalize pixel values to be between 0 and 1.
   - **One-Hot Encoding**: Convert labels to one-hot encoded vectors.

2. **Model Building**:
   - **Model Architecture**: Build a CNN using Keras with layers for convolution, pooling, flattening, and dense connections.
   - **Compiling the Model**: Compile the model with the Adam optimizer and categorical cross-entropy loss.

3. **Model Training and Evaluation**:
   - **Training the Model**: Train the model on the training dataset.
   - **Evaluating the Model**: Evaluate the model on the test dataset to determine its accuracy.

4. **Saving the Trained Model**:
   - **Saving the Model**: Save the trained model in the HDF5 format.

5. **Streamlit Dashboard**:
   - **Setting Up Streamlit**: Configure and initialize the Streamlit app.
   - **Drawing and Predicting Digits**: Allow users to draw digits on a canvas and predict them using the trained model.

## Contact Information
For inquiries and collaborations, feel free to contact:

- **Silvia Dharma**
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/silviadharma)
