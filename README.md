# Iris_Flower_Classification
# 🌸 Iris Flower Classification (Advanced Machine Learning Project)

This project uses the famous **Iris dataset** to classify flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on their petal and sepal measurements.

It includes:
- Data exploration and visualization
- Feature scaling
- Model comparison (Logistic Regression, SVM, Random Forest)
- Cross-validation
- Hyperparameter tuning with `GridSearchCV`
- Confusion matrix visualization
- Model saving & loading with `joblib`

---

## 📂 Dataset

The dataset is loaded directly from an online CSV file:
- [Iris Dataset CSV (GitHub Gist)](https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv)

**Columns:**
- `sepal_length` (cm)
- `sepal_width` (cm)
- `petal_length` (cm)
- `petal_width` (cm)
- `species` (target variable)

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/iris-flower-classification.git
cd iris-flower-classification
