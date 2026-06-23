from re import VERBOSE
import math
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
print("""
Dataset : PakWheels Used-Car Listings
Source  : Scraped from https://www.pakwheels.com (Pakistan's
          largest online car marketplace).
Domain  : Automotive / E-Commerce
-------------------------------------------------------------
Column           Type         Description
-------------------------------------------------------------
nam              Categorical  Car make, model, variant & year
Price            Numerical    Asking price (raw text, e.g. 'PKR 23.9 lacs')
Year             Numerical    Manufacturing year (integer)
Millage          Numerical    Odometer reading (raw text, e.g.'120,000 km')
Fuel             Categorical  Fuel type (Petrol/Diesel/CNG/LPG/Electric)
Transmission     Categorical  Gearbox type (Manual / Automatic)
Province         Categorical  City/Province of the seller
Color            Categorical  Exterior colour
Assembly         Categorical  Local or Imported assembly
EngineCapacity   Numerical    Engine displacement (raw text, e.g. '660 cc')
BodyType         Categorical  Body style (Hatchback/Sedan/SUV.)
AdReference      Numerical    Unique numeric ad ID (dropped - identifier only)
Features         Text         Comma-separated list of car features
OwnerNam         Categorical  Seller name (dropped )
url              Text         Ad URL (dropped - identifier only)
""")

# this function will add a category and assign a label if it wasn't present in the training data
def extend_and_transform(encoder, column):
    unseen = set(column.unique()) - set(encoder.classes_)
    if unseen:
        encoder.classes_ = np.append(encoder.classes_, list(unseen))
    return encoder.transform(column)

def fix_price_col(p):
    """Convert price strings like 'PKR 23.9 lacs' to float (lacs)."""
    if not isinstance(p, str):
        return np.nan
    p = p.lower()
    numbers = re.findall(r'[\d.]+', p)
    if not numbers:
        return np.nan
    number = float(numbers[0])
    if 'crore' in p or 'cr' in p:
        return number * 100   # 1 crore = 100 lacs
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
    """
    if not isinstance(name, str):
        return 'Unknown'
    return name.strip().split()[0]

# preprocessing start here
df = pd.read_csv('data.csv')
print(f"orignal shape : {df.shape}  ({df.shape[0]} rows x {df.shape[1]} columns)")
df.drop(columns=['url', 'OwnerNam', 'AdReference'], inplace=True)
df.rename(columns={'nam': 'Name'}, inplace=True) 

print("\n Dataypes Before conversion:")
print(df.dtypes)

# extracting brand name from Name coloumn
df['Make'] = df['Name'].apply(extract_make)
df['Price'] = df['Price'].apply(fix_price_col)        
df['EngineCapacity'] = df['EngineCapacity'].apply(fix_capacity_col)  
df['Millage'] = df['Millage'].apply(fix_mileage_col)      

#categorical Col
cat_cols = ['Fuel', 'Transmission', 'Assembly', 'BodyType', 'Color', 'Features', 'Make']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

print("\nDatatypes After conversion:")
print(df.dtypes)

print("\nMissing values after conversion of datatypes:")
print(df.isnull().sum())

for col in ['BodyType', 'Color', 'Features']: # fixed mising
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

df['Province'] = df['Province'].str.strip().str.title()
translation_map = {
    'Punjab': 'Lahore',
    'Sindh': 'Karachi',
    'Kpk': 'Peshawar',
    'Balochistan' : 'Quetta',
    'Ict' : 'Islamabad',
    'Isb': 'Islamabad',
    'Khi': 'Karachi',
    'Lhr' : 'Lahore',
    'Fsd' : 'Faisalabad',
    'Mul' : 'Multan',
}
df['City'] = df['Province'].replace(translation_map)
df = df[df['City'] != 'Un-Registered'] # dropping records having Un-Registered city 
df = df[df['Make'] != 'Unknown'] # dropping Make having with Unknown Marked 
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
# df = df[df['Millage'] > 0.0] # atleast more than 0 km driven
# df = df[df['Price'] >= 5.0] # at least price is 5 lacs 
# df = df[df['EngineCapacity'] >= 300]  # cc must be >=300


print("\n--- Outlier Handling Strategy (Using IQR)---")
print("Rows outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are dropped.")
print("Year column excluded: old cars are valid, not errors.")

print("\nDataset shape BEFORE outlier removal:")
print(f"Rows : {df.shape[0]}")
print(f"Cols : {df.shape[1]}")
print(df[['Price', 'Year', 'Millage', 'EngineCapacity']].describe())

# print("\n-----Outliers Detected-----")
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



#Section 4
print("""
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
     - Handles missing values natively.
     - Will be used in Phase 2 to benchmark against Random Forest
       and select the best model for deployment.
""")

# Phase 1 model 
print("\nWait for Random Forest Regressor model to train ")
X = df.drop(columns=['Price', 'Name'])
Y = df['Price']

encoder_fuel  = LabelEncoder()
encoder_transmission = LabelEncoder()
encoder_color = LabelEncoder()
encoder_assembly = LabelEncoder()
encoder_bodytype = LabelEncoder()
encoder_features  = LabelEncoder()
encoder_city = LabelEncoder()
encoder_make = LabelEncoder()   

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
# Make 
X_train['Make'] = encoder_make.fit_transform(X_train['Make'])
X_test['Make']  = extend_and_transform(encoder_make, X_test['Make'])

model=RandomForestRegressor(random_state=42)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("\n--- Random Forest Model Performance ---")
print(f"MAE (Mean Absolute Error) : {mae:.4f} lacs")
print(f"MSE (Mean Squared Error)  : {mse:.4f}")
print(f"RMSE (Root Mean Squared Error)  : {rmse:.4f} lacs")
print(f"R² (Coefficient of Determination): {r2:.4f}")

# Phase 2 model
# pyrefly: ignore [missing-import]
import xgboost as xgb
print("\nPhase 2: XGBoost Regressor ---")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,  # Number of sequential trees
    learning_rate=0.05, # Shrinkage to prevent overfitting
    max_depth=6,  # Maximum depth of each tree
    subsample=0.8,   # Use 80% of rows per tree
    colsample_bytree=0.8, # Use 80% of features per tree
    random_state=42,
    objective='reg:squarederror' # Standard objective function for regression
)
xgb_model.fit(
    X_train, Y_train,
    eval_set=[(X_test, Y_test)],
    verbose=100
)
xgb_pred = xgb_model.predict(X_test)

xgb_mse = mean_squared_error(Y_test, xgb_pred)
xgb_rmse = math.sqrt(xgb_mse)
xgb_mae = mean_absolute_error(Y_test, xgb_pred)
xgb_r2  = r2_score(Y_test, xgb_pred)

print("\n--- XGBoost Model Performance ---")
print(f"MAE (Mean Absolute Error) : {xgb_mae:.4f} lacs")
print(f"MSE (Mean Squared Error) : {xgb_mse:.4f}")
print(f"RMSE(Root Mean Squared Error): {xgb_rmse:.4f} lacs")
print(f"R²(Coefficient of Determination): {xgb_r2:.4f}")


print("\n--- Model Comparison Benchmark ---")
print(f"{'Metric':<12} {'Random Forest':>16} {'XGBoost':>12}")
print(f"{'MAE (lacs)':<12} {mae:>16.4f} {xgb_mae:>12.4f}")
print(f"{'RMSE (lacs)':<12} {rmse:>16.4f} {xgb_rmse:>12.4f}")
print(f"{'R²':<12} {r2:>16.4f} {xgb_r2:>12.4f}")
print("-" * 42)
better = "XGBoost" if xgb_rmse < rmse else "Random Forest"
print(f"Better model (lower RMSE): {better}")


import json
input_json = """
{
    "Name": "Changan Alsvin",
    "Year": 2024,
    "Millage": 35000,
    "Fuel": "Petrol",
    "Transmission": "Automatic",
    "EngineCapacity": 1500,
    "Assembly":"Local",
    "BodyType": "Sedan",
    "Color": "Black",
    "City": "Lahore",
    "Features": "Air Bags,Power Steering,Power Windows"
}
"""

inp = json.loads(input_json)
inp['Make'] = extract_make(inp['Name'])  # extract make from name
input_df = pd.DataFrame([inp]).drop(columns=['Name'])

input_df['Fuel']  = extend_and_transform(encoder_fuel,input_df['Fuel'])
input_df['Transmission'] = extend_and_transform(encoder_transmission, input_df['Transmission'])
input_df['Color']= extend_and_transform(encoder_color,input_df['Color'])
input_df['Assembly']= extend_and_transform(encoder_assembly, input_df['Assembly'])
input_df['BodyType']= extend_and_transform(encoder_bodytype, input_df['BodyType'])
input_df['Features']= extend_and_transform(encoder_features,input_df['Features'])
input_df['City']= extend_and_transform(encoder_city,input_df['City'])
input_df['Make']= extend_and_transform(encoder_make,input_df['Make'])

input_df = input_df[X_train.columns]# align column order with training
predicted_price = xgb_model.predict(input_df)[0]
print(f"\n--- Prediction ---")
print(f"Car: {inp['Name']} ({inp['Year']})")
print(f"Predicted Price : PKR {predicted_price:.2f} lacs")

