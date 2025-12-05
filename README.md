# House Price Prediction with Regression Models

This project demonstrates a basic **regression workflow** for predicting
house prices using multiple features (size, number of rooms, age, distance
to city centre).

Two models are trained and compared:

- **Linear Regression** (with feature scaling)
- **Random Forest Regressor**

The project highlights model evaluation and visualisation.

## Dataset

The dataset is a small synthetic CSV file containing the following columns:

- `size_sq_m` – size of the house in square metres  
- `num_rooms` – number of rooms  
- `age_years` – age of the building in years  
- `distance_to_city_km` – distance to the city centre in kilometres  
- `price_eur` – selling price in euros  

File: `data/house_prices.csv`

## Project Structure

```text
house-price-regression/
│── data/
│   └── house_prices.csv
│── src/
│   └── regression_models.py
│── requirements.txt
└── README.md
