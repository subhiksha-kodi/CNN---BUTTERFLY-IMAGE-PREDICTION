# Butterfly Image Classification

This repository contains a Jupyter Notebook (`code.ipynb`) and dataset for training and evaluating a Convolutional Neural Network (CNN) to classify butterfly species.

---

## 📂 Project Structure
```
.
├── code.ipynb                       # Main notebook for training & evaluation
├── butterfly-image-classification/  # Dataset (unzipped from Kaggle)
└── README.md                        # Project documentation
```

---

## 📊 Dataset
- **Name:** Butterfly Image Classification
- **Source:** [Kaggle - Butterfly Image Classification](https://www.kaggle.com/datasets)
- **Description:** A collection of butterfly images across multiple species for classification tasks.
- **Format:** Directory of images grouped by class labels.
- **Size:** Thousands of images across various butterfly categories.

> Place the dataset folder (`butterfly-image-classification`) in the project root or adjust paths inside `code.ipynb`.

---

## 🚀 Steps to Run

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Typical packages: `tensorflow`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`)*

3. **Prepare the dataset**
   - Download the butterfly dataset from Kaggle (if not already provided).
   - Unzip it into `butterfly-image-classification/`.

4. **Run the notebook**
   ```bash
   jupyter notebook code.ipynb
   ```
   or upload to [Google Colab](https://colab.research.google.com/) and run all cells.

---

## 📊 Model Workflow
- Data preprocessing and augmentation (rescaling, rotation, flips).
- Building a CNN using Keras/TensorFlow.
- Training the model with training and validation splits.
- Plotting:![MODEL ACCURACY AND LOSS](image.png)

---

## 📊 Results / Insights
- The model achieves good generalization on validation data.
- Validation accuracy may be slightly higher than training accuracy due to regularization (dropout, augmentation).
- No significant overfitting observed.

---

## 🛠️ Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn (optional)

Install all dependencies:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## 📜 License
Specify your project license (e.g., MIT, Apache 2.0).

---

## 🙌 Acknowledgments
- [Kaggle - Butterfly Image Classification Dataset](https://www.kaggle.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)