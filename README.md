# CarPrice
Assignment 11.1 What Drives the Price of a Car

# Overview
This project explores a used car dataset: [vehicles.csv](data/vehicles.csv) of 426,000 used cars (sampled from the original Kaggle dataset of 3 million entries) with the goal of identifying the key factors that influence used car prices. The analysis follows the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology to ensure a structured and repeatable approach. 
<p align="center">
  <img src="images/crisp.png" width="400"/><br>
  <em>Figure 1: CRISP-DM framework.  Source: UC Berkeley</em>
</p>

The final deliverable provides actionable insights and recommendations to a client—a used car dealership—on what consumers value most when purchasing a used car.

Here is [the link of the Jupyter Notebook: Car_Price.ipynb](Car_Price.ipynb) with visualizations and probability distributions. It is developed under Google's Colab.

# Methodology (CRISP-DM Framework)

## 1. Business Understanding  
   - **Objective:**  
     - Identify what makes cars more or less expensive.  
     - Translate business problem into analytical questions  
       (e.g., “How do mileage, age, and brand affect price?”)


## 2. Data Understanding

The raw dataset contains **426,000 rows × 18 columns**, meaning it has 426,000 records with 18 features.  
The features include:  

`id`, `region`, `price`, `year`, `manufacturer`, `model`, `condition`, `cylinders`, `fuel`, `odometer`,  
`title_state`, `transmission`, `VIN`, `drive`, `type`, `size`, `paint_color`, and `state`.  

Among them, three are **numeric features**, and the rest are **categorical features**.  

The dataset has several issues such as missing values, extreme numbers, unusually wide ranges, and fabricated-looking records.  
Below is my initial understanding of the data:

a. **Missing values** – Many features contain missing data. A summary graph shows the percentages of missing values for each feature.  
<p align="center">
  <img src="images/missingness_topN.png" width="800"/><br>
  <em>Figure: Top 18 features with missing values</em>
</p>
b. **Zero values in numeric features** – Some numeric features contain a large number of zeros. A summary graph highlights this issue.  
<p align="center">
  <img src="images/zeros_topN_numeric.png" width="800"/><br>
  <em>Figure: Numeric features with missing values</em>
</p>
c. **Car age range** – The dataset includes vehicles ranging from brand new to over 120 years old. Extremely old cars are unrealistic and do not reflect true market prices.  
<p align="center">
  <img src="images/vehicle_distribution.png" width="800"/><br>
  <em>Figure: Vehicle Age Distribution</em>
</p>
d. **Odometer values** – While many cars have reasonable mileage (up to 500K), some records report **1M to even 10M miles**, which are clearly unrealistic.  

e. **Extreme prices** – Over 7% of cars are listed as *free*. On the other hand, some records show extreme prices of **$1M, $10M, or even $100M**, which are almost certainly fabricated.  

f. **Irrelevant features** – Some features, such as `id` and `VIN`, do not provide meaningful information for predicting car prices. 

g. **Severely missing features** – The `size` feature has over **70% missing values**, making it practically unusable.  

h. **Non-standardized text** – Car model names are highly descriptive and inconsistent, making it difficult to reliably identify unique models.  

i. **Redundancy** – Certain features disclose overlapping information (e.g., `region` and `state`), leading to redundancy in the dataset.  

## 3. Data Preparation

In order to make the dataset useful for modeling, several **data cleaning and transformation steps** were performed, including imputation, feature/record removal, and capping of unrealistic values.

### Data Cleaning
- **Feature removal**:  
  - Dropped `id` and `VIN` because they are unrelated to price.  
  - Dropped `size` due to excessive missing values (>70%).  
  - Dropped `model` because the descriptions were inconsistent and not standardized.  

- **Outlier capping**:  
  - Applied caps to extreme values for key numeric features:  
    - **Price**: limited to \$3,000 – \$100,000  
    - **Age**: maximum of 30 years  
    - **Odometer**: maximum of 300,000 miles  

- **Missing value imputation**:  
  - **Categorical features** → imputed with **mode**  
  - **Numeric features** → imputed with **median**  

### Feature Encoding
To improve correlation between categorical features and car price:  
- Instead of One-Hot Encoding (which produces very sparse features), we applied:  
  - **James–Stein Encoding** for high-cardinality categorical features such as `manufacturer`, `state`, and `paint_color`.  
  - **Ordinal encoding** for features with natural order (e.g., `cylinders`, `type`).  

### Feature Selection
Sequential Feature Selection (SFS) was used to identify the most important predictors.  
The top predictors and their corresponding cross-validation performance (negative MSE) are:  

- **Year** (CV neg-MSE = −0.296639)  
- **Cylinders** (CV neg-MSE = −0.233038)  
- **Odometer** (CV neg-MSE = −0.204125)  
- **Drive** (CV neg-MSE = −0.178581)  
- **Fuel type** (CV neg-MSE = −0.159953)  
- **Manufacturer** (CV neg-MSE = −0.150940)  


