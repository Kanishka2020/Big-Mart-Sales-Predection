# Big-Mart-Sales-Predection
This project is about the Big Mart Sales Prediction using Machine Learning with Python. I also used XGBoost Regressor for Prediction.

## Importing the Dependencies or Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

### Mounting the Google Drive
from google.colab import drive
drive.mount('/content/Train.csv')

### Data Collection and Pre-Processing
Then we starts data Collection

#### loading the data from csv file to Pandas DataFrame
big_mart_data = pd.read_csv('//Train.csv')

#### First 5 rows of the dataframe
big_mart_data.head()

<img width="1580" height="319" alt="image" src="https://github.com/user-attachments/assets/6db0cad2-2a60-435e-a462-ec6ea7dd35fc" />

#### Number of data points & number of features
big_mart_data.shape

<img width="130" height="34" alt="image" src="https://github.com/user-attachments/assets/d1b85f83-a0af-4722-9771-116c693ea6f6" />


#### Getting some information about the dataset
big_mart_data.info()

<img width="786" height="381" alt="image" src="https://github.com/user-attachments/assets/18798d6d-d913-4072-87a6-265a8e2710f2" />

#### Categorical Features:
- Item_Identifier
- Item_Fat_Content
- Item_Type
- Outlet_Identifier
- Outlet_Size
- Outlet_Location_Type
- Outlet_Type

#### Checking for missing values
big_mart_data.isnull().sum()

<img width="296" height="483" alt="image" src="https://github.com/user-attachments/assets/ce868f63-e2a2-439c-bd15-b2ebe167dcba" />

#### Handling Missing Values
Mean --> average

Mode --> more repeated value

#### Mean value of "Item_Weight" column
big_mart_data['Item_Weight'].mean()

<img width="355" height="45" alt="image" src="https://github.com/user-attachments/assets/9a90b9f3-f146-4be2-bf39-6e9ef6ca4a6c" />


#### Filling the missing values in "Item_weight column" with "Mean" value

big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

#### Mode of "Outlet_Size" column
big_mart_data['Outlet_Size'].mode()

<img width="177" height="48" alt="image" src="https://github.com/user-attachments/assets/faebbe0d-c8e5-4b3b-86a1-982842c41486" />

#### Filling the missing values in "Outlet_Size" column with Mode
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

print(mode_of_Outlet_size)

<img width="708" height="48" alt="image" src="https://github.com/user-attachments/assets/149d6dcd-cea6-4f9c-b8e7-4d9549f08a7b" />

miss_values = big_mart_data['Outlet_Size'].isnull()

<img width="473" height="288" alt="image" src="https://github.com/user-attachments/assets/f23e5451-0949-489a-8b59-d1083fc36840" />

big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])

#### Checking for missing values
big_mart_data.isnull().sum()

<img width="297" height="267" alt="image" src="https://github.com/user-attachments/assets/8bfd30f8-04f9-4238-bc89-68679f8294e1" />

### Now we perform Data Analysis
big_mart_data.describe()

<img width="821" height="323" alt="image" src="https://github.com/user-attachments/assets/bf66c641-3050-416e-8699-288aa8a0a9d6" />

#### Numerical Features
sns.set()

#### Item_Weight distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()

<img width="1665" height="437" alt="image" src="https://github.com/user-attachments/assets/15a3c935-4c6b-4f44-9e16-c6f44daa965c" />

#### Item Visibility distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Visibility'])
plt.show()

<img width="815" height="473" alt="image" src="https://github.com/user-attachments/assets/e86efbeb-1cc2-4b97-93fd-918ed8bcf485" />

#### Item MRP distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_MRP'])
plt.show()

<img width="497" height="420" alt="image" src="https://github.com/user-attachments/assets/82d3f8e9-f332-4e87-9a37-dc6dd0916afc" />

#### Item_Outlet_Sales distribution
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Outlet_Sales'])
plt.show()

<img width="524" height="428" alt="image" src="https://github.com/user-attachments/assets/a3bd3df7-ca3c-4567-81d1-80af4531ecb3" />

#### Outlet_Establishment_Year column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.show()

<img width="442" height="425" alt="image" src="https://github.com/user-attachments/assets/b0682545-e167-4629-9706-4223141b63f6" />

### Then evaluate Categorical Features
#### Item_Fat_Content column
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()

<img width="519" height="429" alt="image" src="https://github.com/user-attachments/assets/1bf137a7-039c-484b-ad67-06bbcc33588d" />

#### Item_Type column
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()

<img width="1709" height="363" alt="image" src="https://github.com/user-attachments/assets/05e743b9-f48c-4b96-82a9-51ac1c3a8ae4" />

#### Outlet_Size column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
plt.show()

<img width="477" height="438" alt="image" src="https://github.com/user-attachments/assets/f502cc1a-91c0-4354-92b1-dc41170b46e7" />


### Data Pre-preprocessing
big_mart_data.head()

<img width="1576" height="320" alt="image" src="https://github.com/user-attachments/assets/e73f8c07-7126-4618-9a47-0cdef69de0bf" />

big_mart_data['Item_Fat_Content'].value_counts()

<img width="346" height="129" alt="image" src="https://github.com/user-attachments/assets/a88f499a-c9aa-4f3f-97c9-710f9066e33c" />

big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)

big_mart_data['Item_Fat_Content'].value_counts()

<img width="367" height="66" alt="image" src="https://github.com/user-attachments/assets/c7b88005-6f6e-492e-9991-d16f03625b41" />

encoder = LabelEncoder()

big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])

big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])

big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])

big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])

big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])

big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])

big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

big_mart_data.head()

<img width="1575" height="227" alt="image" src="https://github.com/user-attachments/assets/9d93947d-94e2-4508-be46-aa7dc010c582" />

### Splitting features and Target
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']

print(X)

<img width="671" height="282" alt="image" src="https://github.com/user-attachments/assets/a1c5af10-fdfd-49c8-886d-2f2f5026193c" />

print(Y)

<img width="477" height="250" alt="image" src="https://github.com/user-attachments/assets/f454e671-48ac-4375-917e-63a616e93fae" />

### Splitting the data into Training data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

<img width="299" height="41" alt="image" src="https://github.com/user-attachments/assets/5aa54102-6d2d-4898-946c-41feadfb5a82" />

### Machine Learning Model Training

#### XGBoost Regressor

regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

<img width="1101" height="171" alt="image" src="https://github.com/user-attachments/assets/8d61f037-62da-4e9b-a445-01910b750ecb" />

### Model Evaluation
#### prediction on training data
training_data_prediction = regressor.predict(X_train)

#### R squared Value
r2_train = metrics.r2_score(Y_train, training_data_prediction)

print('R Squared value = ', r2_train)

<img width="346" height="35" alt="image" src="https://github.com/user-attachments/assets/2e8d683c-3cbb-47a8-9623-ba6e3ee2fa65" />

#### prediction on test data
test_data_prediction = regressor.predict(X_test)

#### R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)

print('R Squared value = ', r2_test)

<img width="353" height="34" alt="image" src="https://github.com/user-attachments/assets/1dc6c6da-1d91-42aa-a04e-09874cc06a19" />




