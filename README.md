# Pneumonia Detection from Chest X-rays

## Project Overview

This project implements a deep learning model to detect pneumonia from chest X-ray images using the RSNA Pneumonia Detection Challenge dataset. The model uses transfer learning with a pre-trained ResNet50 architecture and is designed to run in Google Colab.

## Dataset Information

The RSNA Pneumonia Detection Challenge dataset contains chest X-ray images labeled for pneumonia presence. The dataset follows this structure:

- `stage_2_train_images/`: Directory containing training DICOM images
- `stage_2_test_images/`: Directory containing test DICOM images
- `stage_2_train_labels.csv`: CSV file with labels for training images
- `stage_2_detailed_class_info.csv`: Detailed class information (optional)

## Setup Instructions

1. **Upload the dataset to Google Drive**
   - Download the RSNA Pneumonia Detection Challenge dataset from Kaggle
   - Create a folder in your Google Drive called `pneumonia_detection`
   - Upload the dataset ZIP file to this folder

2. **Download the trained model**
   - The trained model can be downloaded from Google Drive:  
     [Final Pneumonia Model](https://drive.google.com/file/d/11fyMiUohMB4zz6Mo822uFSe_4mCsf8js/view?usp=sharing)

3. **Run the Colab notebook**
   - Open the provided `pneumonia_detection_notebook.ipynb` file in Google Colab
   - Connect to a GPU runtime (recommended)
   - Run the cells sequentially

## Project Structure

The project follows this workflow:

1. **Environment Setup**
   - Install required libraries
   - Mount Google Drive
   - Configure GPU

2. **Data Preparation**
   - Extract and organize dataset
   - Read and process DICOM images
   - Split data into training and validation sets
   - Create data generators with augmentation

3. **Model Architecture**
   - ResNet50 base model with transfer learning
   - Custom classification layers
   - Two-phase training (frozen base model, then fine-tuning)

4. **Training Pipeline**
   - Configure callbacks for early stopping and model checkpointing
   - Train in two phases (head layers, then fine-tuning)
   - Plot training metrics

5. **Evaluation**
   - Evaluate model on validation set
   - Generate confusion matrix, ROC curve, classification report
   - Visualize sample predictions

6. **Inference**
   - Generate predictions on test set
   - Save results to submission CSV
   - Convert model to TFLite format (optional)

## Model Deployment

The model can be deployed using:
- **Saved Keras Model (`.keras` format)**
- **TFLite Model (`.tflite` for mobile/edge deployment)**
- **Batch Predictions** for test images

## Making Predictions

To make predictions on individual images:
```python
import gdown

# Download model
url = "https://drive.google.com/file/d/11fyMiUohMB4zz6Mo822uFSe_4mCsf8js/view?usp=sharing"
gdown.download(url, "final_pneumonia_model.keras", quiet=False)

def predict_pneumonia(model, image_path):
    """Predict pneumonia on a single image"""
    img = read_dicom_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    prediction = model.predict(img_batch)[0][0]

    return {
        'probability': float(prediction),
        'prediction': 'Pneumonia' if prediction > 0.5 else 'Normal',
        'image': img
    }
```

## Troubleshooting

- **Memory Errors:** Reduce batch size or image dimensions
- **Slow Training:** Ensure GPU runtime is enabled
- **Missing Files:** Check dataset file paths
- **DICOM Issues:** Verify pydicom installation

## References

- [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- [Transfer Learning with ResNet50](https://keras.io/api/applications/resnet/)
- [Deep Learning for Medical Imaging](https://www.tensorflow.org/tutorials/images/classification)

