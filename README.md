# Auto MPG Prediction using Machine Learning

##  Overview

This project predicts a vehicle's fuel efficiency (Miles Per Gallon - MPG) using Machine Learning regression models. The objective is to analyze the relationship between vehicle characteristics and fuel efficiency, then build predictive models capable of estimating MPG for unseen data.

The project includes data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, evaluation, and deployment using Streamlit.

---

##  Dataset

The project uses the **Auto MPG** dataset, which contains various vehicle specifications, including:

* Cylinders
* Displacement
* Horsepower
* Weight
* Acceleration
* Model Year
* Origin

**Target Variable:**

* MPG (Miles Per Gallon)

---

## Project Workflow

1. Data Cleaning

   * Replaced missing values
   * Converted data types
   * Removed unnecessary columns

2. Exploratory Data Analysis (EDA)

   * Feature distributions
   * Correlation heatmap
   * Scatter plots

3. Data Preprocessing

   * Train/Test split
   * Feature scaling using StandardScaler

4. Model Training

   * Linear Regression
   * Support Vector Regression (SVR)

5. Hyperparameter Tuning

   * GridSearchCV

6. Model Evaluation

   * Mean Absolute Error (MAE)
   * Root Mean Squared Error (RMSE)
   * R² Score
   * Cross Validation

7. Deployment

   * Saved the best model using Pickle
   * Developed a Streamlit web application for predictions

---

## Machine Learning Models

* Linear Regression
* Support Vector Regression (SVR)
* GridSearchCV for Hyperparameter Optimization

---

## Evaluation Metrics

The models were evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score
* Cross Validation

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SciPy
* Streamlit
* Pickle

---

## Application Preview

### Home Screen

*(Add your screenshot here)*

### Prediction Result

*(Add your screenshot here)*

---

## Project Structure

```text
auto-mpg-prediction/
│
├── auto_mpg_model.ipynb
├── auto_mpg_model.py
├── app.py
├── auto-mpg.csv
├── requirements.txt
├── README.md
└── screenshots/
```

---

## Run the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Future Improvements

* Add more regression models for comparison.
* Improve the Streamlit interface.
* Perform feature engineering to enhance model performance.
* Deploy the application online.

---

## 👥 Team

This project was developed as a team project by:

* Saja Nashi Al-Fahmi
* Arwa Fahd Al-Roqi
* Lama Saleh Al-Qarni
* Rayana Samer Al-Otaibi

