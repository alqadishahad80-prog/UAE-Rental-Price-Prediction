# 🏠 UAE Rental Price Prediction

## 1. Project Overview

Our project is about predicting rental prices in the UAE using Data Science and Machine Learning. We used a UAE properties dataset to understand rental prices and how different property features affect them.

We used Python in Jupyter Notebook for data cleaning, analysis, and machine learning. We also created a Streamlit application where users can enter property details and see the predicted rental price, rental category, and property group.

**Live Application:**

https://uae-rental-price-prediction-hvb3u8hkcp5fdzp8gkhpha.streamlit.app/

## 2. Business Problem

Rental prices in the UAE are different depending on the location, property size, number of rooms, and property type.

Our goal is to use machine learning to help people understand rental prices and compare properties more easily.

We focused on three tasks:

- **Regression:** Predict the yearly rental price in AED.
- **Classification:** Predict whether the rental category is Low, Medium, or High.
- **Clustering:** Group properties with similar characteristics.

## 3. Team Members and Responsibilities

| Team Member | Responsibilities |
|---|---|
|Shahad Alshehhi| Organized the project, managed GitHub, worked on Streamlit development and deployment, integrated project files, and reviewed the final work. |
|Maryam Alshehhi| Worked on data cleaning, exploratory data analysis, regression modelling, and checking model results. |
|Shaikha Al Ali| Worked on feature engineering, classification, clustering, testing, and project documentation. |

All three members worked together to discuss the results, review the work, and improve the project. Shahad also coordinated the project and checked the final submission.

## 4. Dataset Information

We used the UAE Properties dataset from Kaggle.

**Dataset source:**

https://www.kaggle.com/datasets/azharsaleem/real-estate-goldmine-dubai-uae-rental-market

- Original records: 73,742
- Original attributes: 17
- Cleaned records: 73,727
- Prepared attributes: 27
- Feature-engineered attributes: 31

The dataset includes rental prices, city, property type, bedrooms, bathrooms, furnishing, property size, and location information.

**Licence:** The dataset licence should be checked on its original Kaggle page before redistribution.

## 5. Data Cleaning and Preparation

We used Python and Altair AI Studio to prepare the dataset.

Our cleaning steps included:

1. Checking missing values and duplicate records.
2. Removing duplicate records if found.
3. Removing unnecessary columns.
4. Replacing missing Latitude and Longitude values using the average.
5. Removing records with invalid rental prices.
6. Converting categorical features into numerical values.
7. Keeping Rent and Rent_category for future modelling.
8. Saving the cleaned dataset.

These steps helped us prepare better-quality data for machine learning.

## 6. Exploratory Data Analysis (EDA)

We used Python to understand the dataset and create visualizations.

Our analysis included:

- Rental price distribution.
- Rental categories.
- Rental prices by city.
- Rental prices by property type.
- Property size compared with rental price.
- Correlation between numerical features.

This helped us understand the data before training the models.

## 7. Feature Engineering

We created additional features to improve the information used by our models.

The new features included:

- Sqft_per_bedroom
- Bathrooms_per_bedroom
- Total_rooms
- Is_studio

These features help describe property size and room information.

## 8. Machine Learning Models

We used three machine learning tasks.

### Regression

We used Ridge Regression and Decision Tree Regression to predict yearly rental prices.

We evaluated the models using R², MAE, RMSE, and cross-validation.

### Classification

We used Decision Tree and Random Forest to classify rental prices into Low, Medium, and High.

We evaluated the models using accuracy, precision, recall, F1-score, and a confusion matrix.

### Clustering

We used K-Means clustering to group similar rental properties.

We evaluated the clusters using silhouette scores and cluster analysis.

We also used GridSearchCV for model tuning in our regression and classification notebooks.

## 9. Model Results

Our initial model results were:

| Model | Result |
|---|---|
| Regression | Test R² ≈ 0.133 |
| Classification | Random Forest accuracy ≈ 86.1% |
| Clustering | Silhouette score ≈ 0.322 |

The classification model showed promising results, but regression needs improvement. The clustering model identified three groups of properties.

These are initial evaluation results. More testing is needed before using the models for real rental decisions.

## 10. GitHub Project Structure

Our repository contains the following folders:

**data/** – Original and cleaned datasets.

**notebooks/** – Six Jupyter Notebooks:

1. 01_data_preparation.ipynb
2. 02_EDA.ipynb
3. 03_feature_engineering.ipynb
4. 04_regression.ipynb
5. 05_classification.ipynb
6. 06_clustering.ipynb

**models/** – Saved Regression, Classification, and Clustering models.

**app/** – Streamlit application (`app.py`).

**docs/** – Supporting project documentation.

**requirements.txt** – Python libraries required for the application.

## 11. How to Run the Project

Download or clone the GitHub repository.

Install the required libraries by running:

`pip install -r requirements.txt`

Start the application by running:

`streamlit run app/app.py`

The application loads the saved models and allows users to enter property details and generate predictions.

The notebooks can also be opened in Jupyter Notebook and run in order from 01 to 06 using the required dataset files.

## 12. Streamlit Application

We deployed our application using Streamlit Community Cloud.

Users can enter:

- City and property type.
- Furnishing status.
- Number of bedrooms and bathrooms.
- Property area in square feet.
- Days since listing.
- Geographic coordinates (optional).

The application displays the estimated yearly rental price, rent category, and property cluster.

**Live Streamlit Application:**

https://uae-rental-price-prediction-hvb3u8hkcp5fdzp8gkhpha.streamlit.app/

## 13. AI Tools Used

We used ChatGPT to support Python coding, code organization, debugging, and Streamlit development.

We also used ChatGPT to help prepare the technical documentation.

We ran our notebooks in Jupyter, checked the outputs, uploaded the project files to GitHub, and tested the Streamlit application.

## 14. Project Limitations and Future Improvements

Our models have some limitations. Very high rental prices may affect predictions, and some missing location values were replaced with averages.

In the future, we can improve the project by:

- Testing more machine learning algorithms.
- Improving regression accuracy.
- Using more recent rental data.
- Testing models on new properties.
- Improving the Streamlit application.

Our models are currently used for educational purposes and are not approved for real financial decisions.

## 15. Conclusion

In this project, we used Data Science and Machine Learning to analyze UAE rental properties. We completed data preparation, EDA, feature engineering, modelling, evaluation, and Streamlit deployment.

We also organized our files on GitHub and created a working application that demonstrates our three machine learning tasks.

This project helped us understand how data science can be used to solve business problems in the UAE rental market.
