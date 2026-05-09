
from re import VERBOSE
import math
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import re
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder



print("""
Dataset : PakWheels Used-Car Listings
Source  : Scraped from https://www.pakwheels.com (Pakistan's
          largest online car marketplace).
Domain  : Automotive / E-Commerce

FEATURES
-------------------------------------------------------------
Column           Type         Description
-------------------------------------------------------------
nam              Categorical  Car make, model, variant & year
Price            Numerical    Asking price (raw text, e.g. 'PKR 23.9 lacs')
Year             Numerical    Manufacturing year (integer)
Millage          Numerical    Odometer reading (raw text, e.g. '120,000 km')
Fuel             Categorical  Fuel type (Petrol/Diesel/CNG/LPG/Electric)
Transmission     Categorical  Gearbox type (Manual / Automatic)
Province         Categorical  City/Province of the seller
Color            Categorical  Exterior colour
Assembly         Categorical  Local or Imported assembly
EngineCapacity   Numerical    Engine displacement (raw text, e.g. '660 cc')
BodyType         Categorical  Body style (Hatchback/Sedan/SUV...)
AdReference      Numerical    Unique numeric ad ID (dropped - identifier only)
Features         Text         Comma-separated list of car features
OwnerNam         Categorical  Seller name (dropped - PII)
url              Text         Ad URL (dropped - identifier only)

""")

# this function will add a category and assign a label if it wasn't present in the training data
def extend_and_transform(encoder, column):
    unseen = set(column.unique()) - set(encoder.classes_)
    if unseen:
        encoder.classes_ = np.append(encoder.classes_, list(unseen))
    return encoder.transform(column)

def fix_price_col(p):
    """Convert price strings like 'PKR 23.9 lacs' / '1.2 Crore' to float (lacs)."""
    if not isinstance(p, str):
        return np.nan
    p = p.lower()
    numbers = re.findall(r'[\d.]+', p)
    if not numbers:
        return np.nan

    number = float(numbers[0])
    if 'crore' in p or 'cr' in p:
        return number * 100          # 1 crore = 100 lacs
    return number


def fix_capacity_col(p):
    """Convert engine strings like '660 cc' to float (cc)."""
    if not isinstance(p, str):
        return np.nan
    numbers = re.findall(r'[\d]+', p)
    if not numbers:
        return np.nan
    return float(numbers[0])


def fix_mileage_col(p):
    """Convert mileage strings like '120,000 km' to float (km)."""
    if not isinstance(p, str):
        return np.nan
    numbers = re.findall(r'[\d,]+', p)
    if not numbers:
        return np.nan
    return float(numbers[0].replace(',', ''))


def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"  {col}: {n_out} outliers | valid range [{lower:.1f}, {upper:.1f}]")

    return df[(df[col] >= lower) & (df[col] <= upper)]


def extract_make(name):
    """Extract the car brand/make from the full name string.
    e.g. 'Toyota Corolla GLi 1.3 VVTi 2012' -> 'Toyota'
    """
    if not isinstance(name, str):
        return 'Unknown'
    return name.strip().split()[0]


df = pd.read_csv('data.csv')
print(f"orignal shape : {df.shape}  ({df.shape[0]} rows x {df.shape[1]} columns)")
df.drop(columns=['url', 'OwnerNam', 'AdReference'], inplace=True)
df.rename(columns={'nam': 'Name'}, inplace=True)

# --- Extract Make (brand) from Name ---
# Name is kept for search/lookup; Make is used as a model feature
df['Make'] = df['Name'].apply(extract_make)
print("\nTop 10 car makes in dataset:")
print(df['Make'].value_counts().head(10))

print("\n Dataypes Before conversion:")
print(df.dtypes)


df['Price'] = df['Price'].apply(fix_price_col)        
df['EngineCapacity'] = df['EngineCapacity'].apply(fix_capacity_col)  
df['Millage'] = df['Millage'].apply(fix_mileage_col)      

cat_cols = ['Fuel', 'Transmission', 'Assembly', 'BodyType', 'Color', 'Features', 'Make']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

print("\nDatatypes After conversion:")
print(df.dtypes)

print("\nMissing values after conversion of datatypes:")
print(df.isnull().sum())

for col in ['BodyType', 'Color', 'Features']: # fixing missing
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

df['Province'] = df['Province'].str.strip().str.title()
translation_map = {
    'Punjab'      : 'Lahore',
    'Sindh'       : 'Karachi',
    'Kpk'         : 'Peshawar',
    'Balochistan' : 'Quetta',
    'Ict'         : 'Islamabad',
    'Isb'         : 'Islamabad',
    'Khi'         : 'Karachi',
    'Lhr'         : 'Lahore',
    'Fsd'         : 'Faisalabad',
    'Mul'         : 'Multan',
}
df['City'] = df['Province'].replace(translation_map)
df = df[df['City'] != 'Un-Registered'] # dropping records having Un-Registered city 
df['City'] = df['City'].astype('category')
df.drop(columns=['Province'], inplace=True) # drop province because we have city 

df = df.dropna() # droping null
df.drop_duplicates(inplace=True) # droping duplicates

print("\nMissing values AFTER imputation:")
print(df.isnull().sum())

print("\n--- Missing Value Strategy ---")
print("BodyType/Color/Features , filled with Mode")
print("Converted Province to City using mapping and Dropped province")
print("Remaining NaN rows, dropped ")

# filtering dataset 
df = df[df['Millage'] > 0.0] # atleast more than 0 km driven
df = df[df['Price'] >= 5.0] # at least price is 5 lacs 
df = df[df['EngineCapacity'] >= 300]  # cc must be >=300


print("\n--- Outlier Handling Strategy (Using IQR)---")
print("Rows outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are dropped.")
print("Year column excluded: old cars are valid, not errors.")

print("\nDataset shape BEFORE outlier removal:")
print(f"Rows : {df.shape[0]}")
print(f"Cols : {df.shape[1]}")
print(df[['Price', 'Year', 'Millage', 'EngineCapacity']].describe())

print("\n-----Outliers Detected-----")
numeric_cols = ['Price', 'Millage', 'EngineCapacity'] 
for title, color in [('BEFORE', 'steelblue'), ('AFTER', 'seagreen')]:
    if title == 'AFTER':
        for col in numeric_cols:
            df = remove_outliers_iqr(df, col)   
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Boxplots — {title} Outlier Removal', fontsize=14, fontweight='bold')
    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(y=df[col], ax=ax, color=color)
        ax.set_title(col)
    plt.tight_layout()
    plt.show()


print("\nDataset shape AFTER outlier removal:")
print(f"Rows : {df.shape[0]}")
print(f"Cols : {df.shape[1]}")
print(df[['Price', 'Year', 'Millage', 'EngineCapacity']].describe())

print(f"\nFinal dataset shape : {df.shape}  ({df.shape[0]} rows x {df.shape[1]} columns)")

print("\nColumn data types:")
print(df.dtypes)



# section 3 
# numerical data histogram
plt.hist(df['Price'])
plt.title('Price Distribution')
plt.xlabel('Price in lacs')
plt.ylabel('Frequency')
plt.show()

plt.hist(df['Millage'])
plt.title('Millage Distribution in km')
plt.xlabel('Millage')
plt.ylabel('Frequency')
plt.show()

plt.hist(df['EngineCapacity'])
plt.title('EngineCapacity Distribution')
plt.xlabel('EngineCapacity in cc')
plt.ylabel('Frequency')
plt.show()

plt.hist(df['Year'])
plt.title('Year Distribution')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.show()

# some categorical data pie charts  

#body type
bodytype_count = df['BodyType'].value_counts()
bodytype_count=bodytype_count[0:10]
plt.figure(figsize=(10,8))
plt.pie(bodytype_count.to_list(),labels=list(bodytype_count.index))
plt.legend()
plt.title("BodyType")
plt.show()

# Color
color_count = df['Color'].value_counts()
color_count = color_count[0:10]
plt.figure(figsize=(10, 8))
plt.pie(color_count.to_list(), labels=list(color_count.index))
plt.legend()
plt.title("Color")
plt.show()

# City
city_count = df['City'].value_counts()
city_count = city_count[0:10]
plt.figure(figsize=(10, 8))
plt.pie(city_count.to_list(), labels=list(city_count.index))
plt.legend()
plt.title("City")
plt.show()

# scatter plot between the price and the Millage
plt.scatter(df['Millage'], df['Price'])  
plt.title('Mileage vs Price')
plt.xlabel('Mileage (km)')
plt.ylabel('Price (lacs)')
plt.show()

# pair plot
sample_df = df[['Price', 'Year', 'Millage', 'EngineCapacity']].sample(500, random_state=42)
sns.pairplot(sample_df, 
             plot_kws={'s': 5, 'alpha': 0.3},
             diag_kws={'bins': 20})
plt.show()  



# Section 4
print("""
Dataset Task:
  Predict the selling price (in PKR lacs) of a used car listed
  on PakWheels, based on features such as make/model year,
  mileage, engine capacity, fuel type, transmission, body type,
  assembly, colour, and city.

Models Selected:
  1. Random Forest Regressor
     - An ensemble of decision trees trained on random subsets
       of data and features (bagging).
     - Handles non-linear relationships well.
     - Robust to outliers and noisy features.
     - Provides feature importance scores for interpretability.
     - Suitable here because car prices depend on multiple
       interacting factors in a non-linear way.

  2. XGBoost Regressor (Phase 2)
     - Gradient-boosted ensemble: trees are built sequentially,
       each correcting errors of the previous one.
     - Generally achieves higher accuracy than Random Forest on
       tabular data due to boosting.
     - Built-in L1/L2 regularisation reduces overfitting.
     - Handles missing values natively.
     - Will be used in Phase 2 to benchmark against Random Forest
       and select the best model for deployment.

""")

print("\nWait for model to train ")

# Drop Name (kept only for search/lookup, not a model feature)
# Make (brand) is extracted from Name and used as a feature instead
X = df.drop(columns=['Price', 'Name'])
Y = df['Price']

encoder_fuel         = LabelEncoder()
encoder_transmission = LabelEncoder()
encoder_color        = LabelEncoder()
encoder_assembly     = LabelEncoder()
encoder_bodytype     = LabelEncoder()
encoder_features     = LabelEncoder()
encoder_city         = LabelEncoder()
encoder_make         = LabelEncoder()   # NEW: encode car brand

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# Fuel
X_train['Fuel']= encoder_fuel.fit_transform(X_train['Fuel'])
X_test['Fuel']= extend_and_transform(encoder_fuel, X_test['Fuel'])

# Transmission
X_train['Transmission'] = encoder_transmission.fit_transform(X_train['Transmission'])
X_test['Transmission'] = extend_and_transform(encoder_transmission, X_test['Transmission'])

# Color
X_train['Color']= encoder_color.fit_transform(X_train['Color'])
X_test['Color']= extend_and_transform(encoder_color, X_test['Color'])

# Assembly
X_train['Assembly']= encoder_assembly.fit_transform(X_train['Assembly'])
X_test['Assembly']= extend_and_transform(encoder_assembly, X_test['Assembly'])

# Body Type
X_train['BodyType']= encoder_bodytype.fit_transform(X_train['BodyType'])
X_test['BodyType']= extend_and_transform(encoder_bodytype, X_test['BodyType'])

# Features
X_train['Features']= encoder_features.fit_transform(X_train['Features'])
X_test['Features']= extend_and_transform(encoder_features, X_test['Features'])

# City
X_train['City']= encoder_city.fit_transform(X_train['City'])
X_test['City']= extend_and_transform(encoder_city, X_test['City'])

# Make (NEW)
X_train['Make'] = encoder_make.fit_transform(X_train['Make'])
X_test['Make']  = extend_and_transform(encoder_make, X_test['Make'])


model=RandomForestRegressor(random_state=42)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)

mse  = mean_squared_error(Y_test, y_pred)
rmse = math.sqrt(mse)
r2   = r2_score(Y_test, y_pred)

print("\n--- Random Forest Model Performance ---")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error) : {rmse:.4f} lacs")
print(f"R^2 score (Coefficient of Determination): {r2:.4f}")


# I. DATASET BEING USED
# ─────────────────────
# Dataset : PakWheels Used-Car Listings
# Source  : Scraped from https://www.pakwheels.com — Pakistan's largest online
#           automotive marketplace.
# Domain  : Automotive / E-Commerce
# Size    : Several thousand real-world used-car advertisements.
#
# The dataset contains the following key features:
#   • Name            – Car make, model, variant & manufacturing year (string)
#   • Make            – Extracted car brand/manufacturer (e.g. Toyota, Honda)
#   • Price           – Asking price in PKR (converted to numeric lacs)
#   • Year            – Manufacturing year (integer)
#   • Millage         – Odometer reading (converted to numeric km)
#   • Fuel            – Fuel type: Petrol / Diesel / CNG / LPG / Electric
#   • Transmission    – Gearbox type: Manual / Automatic
#   • Province/City   – Seller's location (standardised to city names)
#   • Color           – Exterior colour of the vehicle
#   • Assembly        – Local or Imported assembly
#   • EngineCapacity  – Engine displacement (converted to numeric cc)
#   • BodyType        – Body style: Hatchback / Sedan / SUV / etc.
#   • Features        – Comma-separated list of additional car features
#
# Note on Name vs Make:
#   The full Name column (e.g. "Toyota Corolla GLi 1.3 VVTi 2012") is kept in
#   the dataframe for search/lookup purposes (user can type "Corolla" to find
#   matching listings). However it is NOT fed directly to the model because it
#   has extremely high cardinality (thousands of unique values). Instead, Make
#   (the first word / brand) is extracted and label-encoded as a feature, which
#   captures the brand-level price signal (e.g. Toyota vs Suzuki vs Mercedes).
#
# Preprocessing steps applied:
#   – Removed identifier/PII columns (url, OwnerNam, AdReference)
#   – Extracted Make (brand) from Name
#   – Converted Price, Millage, and EngineCapacity from raw text to floats
#   – Standardised Province values to City names using a mapping dictionary
#   – Imputed missing categorical values (BodyType, Color, Features) with mode
#   – Dropped remaining null rows and duplicate records
#   – Filtered unrealistic records (Price < 5 lacs, Mileage = 0, CC < 300)
#   – Removed outliers in Price, Millage, and EngineCapacity using the IQR method
#
# Target Variable : Price (continuous numeric — regression task)
# Task            : Predict the selling price (in PKR lacs) of a used car
#                   based on its features.
#
# ─────────────────────────────────────────────────────────────────────────────

# Phase 2 

# pyrefly: ignore [missing-import]
import xgboost as xgb
print("\n--- Training Phase 2: XGBoost Regressor ---")

# Initialize the XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=500,           # Number of sequential trees
    learning_rate=0.05,         # Shrinkage to prevent overfitting
    max_depth=6,                # Maximum depth of each tree
    subsample=0.8,              # Use 80% of rows per tree
    colsample_bytree=0.8,       # Use 80% of features per tree
    random_state=42,
    objective='reg:squarederror' # Standard objective function for regression
)

# Train the model
xgb_model.fit(
    X_train, Y_train,
    eval_set=[(X_test, Y_test)],
    verbose=100
)

# Make predictions
xgb_pred = xgb_model.predict(X_test)

# Evaluate performance
xgb_mse  = mean_squared_error(Y_test, xgb_pred)
xgb_rmse = math.sqrt(xgb_mse)
xgb_r2   = r2_score(Y_test, xgb_pred)

print("\n--- XGBoost Model Performance ---")
print(f"MSE (Mean Squared Error): {xgb_mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {xgb_rmse:.4f} lacs")
print(f"R^2 score (Coefficient of Determination): {xgb_r2:.4f}")

# Side-by-side benchmark
print("\n--- Model Comparison Benchmark ---")
print(f"Random Forest RMSE : {rmse:.4f} lacs")
print(f"XGBoost RMSE       : {xgb_rmse:.4f} lacs")

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH FEATURE
# User can type a partial car name (e.g. "corolla") and see matching listings
# ─────────────────────────────────────────────────────────────────────────────

def search_cars(query, dataframe, top_n=10):
    """Search for cars by partial name match (case-insensitive)."""
    mask = dataframe['Name'].str.contains(query, case=False, na=False)
    results = dataframe[mask][['Name', 'Year', 'Millage', 'EngineCapacity',
                                'Fuel', 'Transmission', 'City', 'Price']].head(top_n)
    if results.empty:
        print(f"  No listings found matching '{query}'")
    else:
        print(results.to_string(index=False))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FEATURE  (uses XGBoost — best performing model)
# User fills a dictionary with car details → gets predicted price in lacs
# ─────────────────────────────────────────────────────────────────────────────

# Pick the best model based on benchmark
best_model    = xgb_model if xgb_rmse < rmse else model
best_model_nm = "XGBoost" if xgb_rmse < rmse else "Random Forest"
print(f"\nBest model selected for prediction: {best_model_nm}")

def predict_price(car_details: dict) -> float:
    """
    Predict car price (in PKR lacs) from a dictionary of car attributes.

    Required keys:
        make          – str  : Car brand, e.g. 'Toyota', 'Honda', 'Suzuki'
        year          – int  : Manufacturing year, e.g. 2018
        mileage       – float: Odometer in km, e.g. 80000
        engine_cc     – float: Engine capacity in cc, e.g. 1300
        fuel          – str  : 'Petrol', 'Diesel', 'Hybrid', 'Electric', 'CNG', 'LPG'
        transmission  – str  : 'Manual' or 'Automatic'
        assembly      – str  : 'Local' or 'Imported'
        body_type     – str  : 'Sedan', 'Hatchback', 'SUV', 'Crossover', etc.
        color         – str  : e.g. 'White', 'Black', 'Silver'
        features      – str  : comma-separated features string (can leave as '')
        city          – str  : e.g. 'Lahore', 'Karachi', 'Islamabad'

    Returns:
        Predicted price as float (PKR lacs)
    """

    def safe_encode(encoder, value, col_name):
        """Encode a single value, falling back to 0 if unseen."""
        val = str(value).strip()
        if val not in encoder.classes_:
            print(f"  [Warning] '{val}' not seen during training for '{col_name}'. "
                  f"Using closest known value.")
            # Use the most common class (index 0 after fit) as fallback
            val = encoder.classes_[0]
        return int(encoder.transform([val])[0])

    row = {
        'Year'           : int(car_details['year']),
        'Millage'        : float(car_details['mileage']),
        'EngineCapacity' : float(car_details['engine_cc']),
        'Fuel'           : safe_encode(encoder_fuel,         car_details['fuel'],         'Fuel'),
        'Transmission'   : safe_encode(encoder_transmission, car_details['transmission'], 'Transmission'),
        'Color'          : safe_encode(encoder_color,        car_details['color'],        'Color'),
        'Assembly'       : safe_encode(encoder_assembly,     car_details['assembly'],     'Assembly'),
        'BodyType'       : safe_encode(encoder_bodytype,     car_details['body_type'],    'BodyType'),
        'Features'       : safe_encode(encoder_features,     car_details.get('features', ''), 'Features'),
        'City'           : safe_encode(encoder_city,         car_details['city'],         'City'),
        'Make'           : safe_encode(encoder_make,         car_details['make'],         'Make'),
    }

    input_df = pd.DataFrame([row])
    # Ensure column order matches training
    input_df = input_df[X_train.columns]

    predicted = best_model.predict(input_df)[0]
    return round(float(predicted), 2)


# ── EXAMPLE USAGE ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("         CAR PRICE PREDICTION — EXAMPLE")
print("="*60)

# Example 1: Search for Corolla listings
print("\n[SEARCH] Listings matching 'Corolla':")
search_cars('Corolla', df, top_n=5)

# Example 2: Predict price using a dictionary
my_car = {
    'make'        : 'Toyota',
    'year'        : 2018,
    'mileage'     : 80000,
    'engine_cc'   : 1300,
    'fuel'        : 'Petrol',
    'transmission': 'Manual',
    'assembly'    : 'Local',
    'body_type'   : 'Sedan',
    'color'       : 'White',
    'features'    : "['ABS', 'Air Conditioning', 'Power Steering']",
    'city'        : 'Lahore',
}

predicted_price = predict_price(my_car)
print(f"\n[PREDICT] Input car details:")
for k, v in my_car.items():
    print(f"  {k:15s}: {v}")
print(f"\n  --> Predicted Price ({best_model_nm}): PKR {predicted_price} lacs")

print("\n" + "="*60)

# ── TO USE IN YOUR OWN CODE ────────────────────────────────────────────────
# Just fill in the dictionary and call predict_price():
#
# my_car = {
#     'make'        : 'Honda',
#     'year'        : 2020,
#     'mileage'     : 45000,
#     'engine_cc'   : 1500,
#     'fuel'        : 'Petrol',
#     'transmission': 'Automatic',
#     'assembly'    : 'Local',
#     'body_type'   : 'Sedan',
#     'color'       : 'Black',
#     'features'    : "['ABS', 'Air Bags', 'Air Conditioning']",
#     'city'        : 'Karachi',
# }
# price = predict_price(my_car)
# print(f"Predicted price: PKR {price} lacs")
