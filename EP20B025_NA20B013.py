import sklearn
import pandas as pd
from matplotlib import pyplot as plt
import csv
import numpy as np
# import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('../data/train_data.csv')
test = pd.read_csv('../data/sample_submission_5.csv')

data = pd.concat([train,test], axis = 0)
y = data['rating_score']
data.drop(['rating_score'],axis = 1,inplace = True)

customer_id,booking_status,booking_create_timestamp,booking_approved_at,booking_checking_customer_date = {},{},{},{},{}
with open('../data/bookings.csv','r') as readfile:
  reader = csv.reader(readfile)
  next(reader)
  for rows in reader:
    customer_id[rows[0]] = rows[1]
    booking_status[rows[0]] = rows[2]
    booking_create_timestamp[rows[0]] = rows[3]
    booking_approved_at[rows[0]] = rows[4]
    booking_checking_customer_date[rows[0]] = rows[5]

data['customer_id'] = data['booking_id'].apply(lambda x: customer_id.get(x))
data['booking_status'] = data['booking_id'].apply(lambda x: booking_status.get(x))
data['booking_create_timestamp'] = data['booking_id'].apply(lambda x: booking_create_timestamp.get(x))
data['booking_approved_at'] = data['booking_id'].apply(lambda x: booking_approved_at.get(x))
data['booking_checking_customer_date'] = data['booking_id'].apply(lambda x: booking_checking_customer_date.get(x))

hotel_id,seller_agent_id,booking_expiry_date,price,agent_fees = {},{},{},{},{}
with open('../data/bookings_data.csv','r') as readfile:
  reader = csv.reader(readfile)
  next(reader)
  for rows in reader:
    hotel_id[rows[0]] = rows[2]
    seller_agent_id[rows[0]] = rows[3]
    booking_expiry_date[rows[0]] = rows[4]
    price[rows[0]] = rows[5]
    agent_fees[rows[0]] = rows[6]

#booking sequence id seems irrelevant
data['hotel_id'] = data['booking_id'].apply(lambda x: hotel_id.get(x))
data['seller_agent_id'] = data['booking_id'].apply(lambda x: seller_agent_id.get(x))
data['booking_expiry_date'] = data['booking_id'].apply(lambda x: booking_expiry_date.get(x))
data['price'] = data['booking_id'].apply(lambda x: price.get(x))
data['agent_fees'] = data['booking_id'].apply(lambda x: agent_fees.get(x))

customer_unique_id,country = {},{}
with open('../data/customer_data.csv','r') as readfile:
  reader = csv.reader(readfile)
  next(reader)
  for rows in reader:
    customer_unique_id[rows[0]] = rows[1]
    country[rows[0]] = rows[2]

data['customer_unique_id'] = data['customer_id'].apply(lambda x: customer_unique_id.get(x))
data['country'] = data['customer_id'].apply(lambda x: country.get(x))

hotel_category,hotel_name_length,hotel_description_length,hotel_photos_qty = {},{},{},{}
with open('../data/hotels_data.csv','r') as readfile:
  reader = csv.reader(readfile)
  next(reader)
  for rows in reader:
    hotel_category[rows[0]] = rows[1]
    hotel_name_length[rows[0]] = rows[2]
    hotel_description_length[rows[0]] = rows[3]
    hotel_photos_qty[rows[0]] = rows[4]

data['hotel_category'] = data['hotel_id'].apply(lambda x: hotel_category.get(x))
data['hotel_name_length'] = data['hotel_id'].apply(lambda x: hotel_name_length.get(x))
data['hotel_description_length'] = data['hotel_id'].apply(lambda x: hotel_description_length.get(x))
data['hotel_photos_qty'] = data['hotel_id'].apply(lambda x: hotel_photos_qty.get(x))

payment_sequential,payment_type,payment_installments,payment_values = {},{},{},{}
with open('../data/payments_data.csv','r') as readfile:
  reader = csv.reader(readfile)
  next(reader)
  for rows in reader:
    payment_sequential[rows[0]] = rows[1]
    payment_type[rows[0]] = rows[2]
    payment_installments[rows[0]] = rows[3]
    payment_values[rows[0]] = rows[4]

data['payment_sequential'] = data['booking_id'].apply(lambda x: payment_sequential.get(x))
data['payment_type'] = data['booking_id'].apply(lambda x: payment_type.get(x))
data['payment_installments'] = data['booking_id'].apply(lambda x: payment_installments.get(x))
data['payment_values'] = data['booking_id'].apply(lambda x: payment_values.get(x))

# date time features

data['booking_create_timestamp'] = pd.to_datetime(data['booking_create_timestamp'])
data['booking_approved_at'] = pd.to_datetime(data['booking_approved_at'])
data['booking_checking_customer_date'] = pd.to_datetime(data['booking_checking_customer_date'])
data['booking_expiry_date'] = pd.to_datetime(data['booking_expiry_date'])

data['booking_approved_diff'] = (data['booking_approved_at'] - data['booking_create_timestamp'])/pd.Timedelta(seconds = 1)
data['booking_checking_diff'] = (data['booking_checking_customer_date'] - data['booking_approved_at'])/pd.Timedelta(seconds = 1)
data['booking_expiry_diff'] = (data['booking_expiry_date'] - data['booking_checking_customer_date'])/pd.Timedelta(seconds = 1)

months_in_year = 12
seconds_in_day = 24*60*60

# extracting days,months,year,seconds of the day for booking_create_timestamp and creating cyclic features
data['booking_create_timestamp_year'] = data['booking_create_timestamp'].dt.year
data['booking_create_timestamp_month'] = data['booking_create_timestamp'].dt.month

data['booking_create_timestamp_month_sin'] = np.sin(2*np.pi*data.booking_create_timestamp_month/months_in_year)
data['booking_create_timestamp_month_cos'] = np.cos(2*np.pi*data.booking_create_timestamp_month/months_in_year)

data['booking_create_timestamp_day'] = data['booking_create_timestamp'].dt.day

data['booking_create_timestamp_seconds'] = ( data['booking_create_timestamp'].dt.hour * 3600 ) + ( data['booking_create_timestamp'].dt.minute * 60 ) + data['booking_create_timestamp'].dt.second

data['booking_create_timestamp_seconds_sin'] = np.sin(2*np.pi*data.booking_create_timestamp_seconds/seconds_in_day)
data['booking_create_timestamp_seconds_cos'] = np.cos(2*np.pi*data.booking_create_timestamp_seconds/seconds_in_day)


# extracting days,months,year,seconds of the day for booking_approved_at and creating cyclic features
data['booking_approved_at_year'] = data['booking_approved_at'].dt.year
data['booking_approved_at_month'] = data['booking_approved_at'].dt.month

data['booking_approved_at_month_sin'] = np.sin(2*np.pi*data.booking_approved_at_month/months_in_year)
data['booking_approved_at_month_cos'] = np.cos(2*np.pi*data.booking_approved_at_month/months_in_year)

data['booking_approved_at_day'] = data['booking_approved_at'].dt.day

data['booking_approved_at_seconds'] = ( data['booking_approved_at'].dt.hour * 3600 ) + ( data['booking_approved_at'].dt.minute * 60 ) + data['booking_approved_at'].dt.second

data['booking_approved_at_seconds_sin'] = np.sin(2*np.pi*data.booking_approved_at_seconds/seconds_in_day)
data['booking_approved_at_seconds_cos'] = np.cos(2*np.pi*data.booking_approved_at_seconds/seconds_in_day)


# extracting days,months,year,seconds of the day for booking_checking_customer_date and creating cyclic features
data['booking_checking_customer_date_year'] = data['booking_checking_customer_date'].dt.year
data['booking_checking_customer_date_month'] = data['booking_checking_customer_date'].dt.month

data['booking_checking_customer_date_month_sin'] = np.sin(2*np.pi*data.booking_checking_customer_date_month/months_in_year)
data['booking_checking_customer_date_month_cos'] = np.cos(2*np.pi*data.booking_checking_customer_date_month/months_in_year)

data['booking_checking_customer_date_day'] = data['booking_checking_customer_date'].dt.day

data['booking_checking_customer_date_seconds'] = ( data['booking_checking_customer_date'].dt.hour * 3600 ) + ( data['booking_checking_customer_date'].dt.minute * 60 ) + data['booking_checking_customer_date'].dt.second

data['booking_checking_customer_date_seconds_sin'] = np.sin(2*np.pi*data.booking_checking_customer_date_seconds/seconds_in_day)
data['booking_checking_customer_date_seconds_cos'] = np.cos(2*np.pi*data.booking_checking_customer_date_seconds/seconds_in_day)


# extracting days,months,year,seconds of the day for booking_expiry_date and creating cyclic features
data['booking_expiry_date_year'] = data['booking_expiry_date'].dt.year
data['booking_expiry_date_month'] = data['booking_expiry_date'].dt.month

data['booking_expiry_date_month_sin'] = np.sin(2*np.pi*data.booking_expiry_date_month/months_in_year)
data['booking_expiry_date_month_cos'] = np.cos(2*np.pi*data.booking_expiry_date_month/months_in_year)

data['booking_expiry_date_day'] = data['booking_expiry_date'].dt.day

data['booking_expiry_date_seconds'] = ( data['booking_expiry_date'].dt.hour * 3600 ) + ( data['booking_expiry_date'].dt.minute * 60 ) + data['booking_expiry_date'].dt.second

data['booking_expiry_date_seconds_sin'] = np.sin(2*np.pi*data.booking_expiry_date_seconds/seconds_in_day)
data['booking_expiry_date_seconds_cos'] = np.cos(2*np.pi*data.booking_expiry_date_seconds/seconds_in_day)

data.drop(['booking_create_timestamp','booking_approved_at','booking_checking_customer_date','booking_expiry_date',
            'booking_create_timestamp_seconds','booking_create_timestamp_month',
            'booking_approved_at_seconds','booking_approved_at_month',
            'booking_checking_customer_date_seconds','booking_checking_customer_date_month',
            'booking_expiry_date_seconds','booking_expiry_date_month'],axis = 1, inplace = True)


numerical_cols = ['price', 'agent_fees', 'hotel_category','hotel_name_length', 'hotel_description_length', 'hotel_photos_qty',
                 'payment_sequential', 'payment_installments', 'payment_values','booking_create_timestamp_year', 'booking_create_timestamp_month_sin',
       'booking_create_timestamp_month_cos', 'booking_create_timestamp_day',
       'booking_create_timestamp_seconds_sin',
       'booking_create_timestamp_seconds_cos', 'booking_approved_at_year',
       'booking_approved_at_month_sin', 'booking_approved_at_month_cos',
       'booking_approved_at_day', 'booking_approved_at_seconds_sin',
       'booking_approved_at_seconds_cos',
       'booking_checking_customer_date_year',
       'booking_checking_customer_date_month_sin',
       'booking_checking_customer_date_month_cos',
       'booking_checking_customer_date_day',
       'booking_checking_customer_date_seconds_sin',
       'booking_checking_customer_date_seconds_cos',
       'booking_expiry_date_year', 'booking_expiry_date_month_sin',
       'booking_expiry_date_month_cos', 'booking_expiry_date_day',
       'booking_expiry_date_seconds_sin', 'booking_expiry_date_seconds_cos', 'booking_approved_diff', 'booking_checking_diff', 'booking_expiry_diff' ]

data[numerical_cols] = data[numerical_cols].apply(pd.to_numeric)

#pure num features
# data['agent_fraction'] = data['agent_fees'] / data['price']
# data['log_hotel_category'] = data['hotel_category'].apply(np.log1p)
# data['hotel_descbylen'] = data['hotel_description_length'] / data['hotel_name_length']
# data['photos_sq'] = data['hotel_photos_qty'] * data['hotel_photos_qty']
# data['avg_payment_per_install'] = data['payment_values'] / data['payment_installments']

#aggregated categorical features
data['customer_pay'] = ( data.groupby("customer_unique_id")["payment_values"].transform("mean") )
data['hotelroom'] = ( data.groupby("hotel_id")["price"].transform("mean") )
data['avgsellerfees'] = ( data.groupby("seller_agent_id")["agent_fees"].transform("mean") )
data['tourist'] = ( data.groupby("country")["customer_unique_id"].transform("count")/data.customer_unique_id.count() )

new_num_cols = ['customer_pay', 'hotelroom', 'avgsellerfees', 'tourist']

data[new_num_cols] = data[new_num_cols].apply(pd.to_numeric)

numerical_cols = numerical_cols+new_num_cols

numerical_cols.remove('hotel_category')
data.drop(['hotel_category'],axis = 1,inplace = True)

categorical_cols = list(set(data.columns) - set(numerical_cols))
print(categorical_cols)

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),   
     ('oh', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer( transformers=[('num', numerical_transformer, numerical_cols),('cat', categorical_transformer, categorical_cols)])

final = pd.DataFrame(preprocessor.fit_transform(data,y))
final.columns = numerical_cols + categorical_cols

cluster_cols = ['hotel_id', 'customer_unique_id', 'seller_agent_id', 'country']

final["Cluster"] = KMeans(n_clusters=10).fit_predict(final[cluster_cols])
final["Cluster"] = final["Cluster"].astype("object")

y = train['rating_score']

train_data = final.iloc[:50000,:]
test_data = final.iloc[50000:,:]

#SCORES
n_estimators = [200,250,300,350,400]
param_grid = {"n_estimators":n_estimators}
model = RandomForestClassifier(random_state = 0)

scorer = make_scorer(mse,greater_is_better = False)
search =  HalvingGridSearchCV(estimator=model,cv = 4,scoring = scorer, param_grid=param_grid, factor=2).fit(train_data,y)

search.best_params_

search.cv_results_

preds = search.best_estimator_.predict(test_data)
# Save test predictions to file
output = pd.DataFrame({'booking_id': test['booking_id'],
                       'rating_score': preds})
output.to_csv('../output/EP20B025_NA20B013.csv', index=False)