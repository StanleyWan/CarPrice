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


## Data Understanding

The raw dataset contains **426,000 rows × 18 columns**, meaning it has 426,000 records with 18 features.  
The features include:  

`id`, `region`, `price`, `year`, `manufacturer`, `model`, `condition`, `cylinders`, `fuel`, `odometer`,  
`title_state`, `transmission`, `VIN`, `drive`, `type`, `size`, `paint_color`, and `state`.  

Among them, three are **numeric features**, and the rest are **categorical features**.  

The dataset has several issues such as missing values, extreme numbers, unusually wide ranges, and fabricated-looking records.  
Below is my initial understanding of the data:

1. **Missing values** – Many features contain missing data. A summary table shows both the counts and percentages of missing values for each feature.  
2. **Zero values in numeric features** – Some numeric features contain a large number of zeros. A summary table highlights this issue.  
3. **Car age range** – The dataset includes vehicles ranging from brand new to over 120 years old. Extremely old cars are unrealistic and do not reflect true market prices.  
4. **Odometer values** – While many cars have reasonable mileage (up to 500K), some records report **1M to even 10M miles**, which are clearly unrealistic.  
5. **Extreme prices** – Over 7% of cars are listed as *free*. On the other hand, some records show extreme prices of **$1M, $10M, or even $100M**, which are almost certainly fabricated.  
6. **Irrelevant features** – Some features, such as `id` and `VIN`, do not provide meaningful information for predicting car prices.  
7. **Severely missing features** – The `size` feature has over **70% missing values**, making it practically unusable.  
8. **Non-standardized text** – Car model names are highly descriptive and inconsistent, making it difficult to reliably identify unique models.  
9. **Redundancy** – Certain features disclose overlapping information (e.g., `region` and `state`), leading to redundancy in the dataset.  

•	Raw dataset: 426,000 rows × 18 columns.
•	Features include id, region, price, year, manufacturer, model, condition, cylinders, fuel, odometer, title_state, transmission, VIN, drive, type, size, paint color, and state. 
•	Some features contain large amount of missing item(figure), like size (70%)(table)
•	Some features are price unrelated like id,VIN.
•	Some features are overlapped like region and state
•	3 numeric features and 15 categorical features
•	extreme price records exist. 34930 rocords are $0-$9 and 21 records are more than $100M. (outlier)
•	the long range in year records. It comes from 1900 to 2022 (graph)
•	4931 records of odometer is from 0-9 miles and 638 records exceed 1M miles(outlier
•	the name of car model is vague and hard to classify.(example)
•	Contain some funny records, such as free for a new car etc
