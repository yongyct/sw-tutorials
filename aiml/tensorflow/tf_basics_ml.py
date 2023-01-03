
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, mean_squared_error as mse

import tensorflow as tf


# # Regression
# 
# => y = mx + c
# where m = 0.5, c = 5

# ## Without tf.estimator API

# In[47]:


### Prepare data ###
x_data = np.linspace(0, 10, 1000000)
noise = np.random.randn(len(x_data))
y_data = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['x'])
y_df = pd.DataFrame(data=y_data, columns=['y'])

df = pd.concat([x_df, y_df], axis=1)

df.sample(n=250).plot(x='x', y='y', kind='scatter')


# In[48]:


batch_size = 8


# In[49]:


### Define tf.Variable
m = tf.Variable(np.random.randn(1).item())
c = tf.Variable(np.random.randn(1).item())

### Define tf.Placeholder
xph = tf.placeholder(dtype=tf.float32, shape=[batch_size])
yph = tf.placeholder(dtype=tf.float32, shape=[batch_size])

### Define tf ops
y_model = m * xph + c

### Define loss function, a subset of tf ops
error = tf.reduce_sum(tf.square(yph - y_model))

### Define optimizer, & train ops
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)    # --> this appears to loop the backprop


# In[50]:


get_ipython().run_cell_magic('time', '', '### Start global variables initializer\nwith tf.Session() as sess:\n    sess.run(tf.global_variables_initializer())\n    n_batches = 5000\n    \n    for i in range(n_batches):\n        rand_int = np.random.randint(len(x_data), size=batch_size)\n        feed_dict = {xph:x_data[rand_int], yph:y_data[rand_int]}\n        sess.run(train, feed_dict=feed_dict)\n    \n    final_m, final_c = sess.run((m, c))')


# In[53]:


y_hat = final_m * x_data + final_c
final_m, final_c


# In[54]:


df.sample(250).plot(x='x', y='y', kind='scatter')
plt.plot(x_data, y_hat, 'r')


# ## With tf.estimator API

# In[ ]:


feat_cols = [ tf.feature_column.numeric_column(key='x', shape=[1]) ]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.3)


# In[31]:


input_func = tf.estimator.inputs.numpy_input_fn(
    x={'x':x_train}, 
    y=y_train, 
    batch_size=batch_size,
    num_epochs=None, 
    shuffle=True
)

train_input_func = tf.estimator.inputs.numpy_input_fn(
    x={'x':x_train}, 
    y=y_train, 
    batch_size=batch_size,
    num_epochs=1000, 
    shuffle=False
)

eval_input_func = tf.estimator.inputs.numpy_input_fn(
    x={'x':x_eval},
    y=y_eval,
    batch_size=batch_size,
    num_epochs=1000,
    shuffle=False
)


# In[ ]:


estimator.train(input_fn=input_func, steps=1000)


# In[34]:


train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)


# In[35]:


eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)


# In[36]:


print(train_metrics)
print(eval_metrics)


# In[37]:


pred_data = np.linspace(0, 10, 10)
predict_input_func = tf.estimator.inputs.numpy_input_fn(x={'x':pred_data}, shuffle=False)
list(estimator.predict(input_fn=predict_input_func))


# In[40]:


predictions = []
for pred in estimator.predict(input_fn=predict_input_func):
    predictions.append(pred['predictions'])
    
df.sample(n=250).plot(x='x', y='y', kind='scatter')
plt.plot(pred_data, predictions, 'r')


# # Classification

# In[9]:


diabetes = pd.read_csv('data/pima-indians-diabetes.csv')
diabetes.head()


# In[10]:


cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 
                'Blood_pressure', 'Triceps', 
                'Insulin', 'BMI', 'Pedigree']


# In[11]:


### Perform Normalization on quantitative columns
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
diabetes.head()


# In[14]:


num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list(key='Group', vocabulary_list=['A', 'B', 'C', 'D'])
# If number of distinct groups/categories is unknown, can set a "max" threshold and let tf take care
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket(keys='Group', hash_bucket_size=10)

age_bucket = tf.feature_column.bucketized_column(source_column=age, boundaries=[20, 30, 40, 50, 60, 70, 80])


# In[13]:


diabetes['Age'].hist(bins=20)


# In[15]:


feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, 
             diabetes_pedigree, assigned_group, age_bucket]


# In[16]:


x_data = diabetes.drop('Class', axis=1)
y_data = diabetes['Class']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)


# In[21]:


input_func = tf.estimator.inputs.pandas_input_fn(
    x=x_train, 
    y=y_train, 
    batch_size=10, 
    num_epochs=1000, 
    shuffle=True
)

eval_input_func = tf.estimator.inputs.pandas_input_fn(
    x=x_test, 
    y=y_test, 
    batch_size=10, 
    num_epochs=1, 
    shuffle=False
)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)


# In[22]:


model.train(input_fn=input_func, steps=1000)


# In[23]:


results = model.evaluate(input_fn=eval_input_func)


# In[24]:


results


# In[26]:


pred_input_func = tf.estimator.inputs.pandas_input_fn(
    x=x_test,
    batch_size=10,
    num_epochs=1,
    shuffle=False
)

predictions = model.predict(input_fn=pred_input_func)
pred_list = list(predictions)


# In[37]:


y_test_pred = []
for prediction in model.predict(input_fn=pred_input_func):
    y_test_pred.append(prediction['class_ids'])
y_test_pred[:5]


# In[40]:


### DNN Model requires embedded group col instead
embedded_group_col = tf.feature_column.embedding_column(
    categorical_column=assigned_group, 
    dimension=4
)
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, 
             diabetes_pedigree, embedded_group_col, age_bucket]

input_func = tf.estimator.inputs.pandas_input_fn(
    x=x_train, 
    y=y_train, 
    batch_size=10, 
    num_epochs=1000, 
    shuffle=True
)

dnn_model = tf.estimator.DNNClassifier(
    hidden_units=[10, 10, 10], 
    feature_columns=feat_cols, 
    n_classes=2
)


# In[41]:


dnn_model.train(input_fn=input_func, steps=1000)


# In[42]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(
    x=x_test, 
    y=y_test, 
    batch_size=10, 
    num_epochs=1000, 
    shuffle=False
)


# In[44]:


results = dnn_model.evaluate(input_fn=eval_input_func, steps=1000)


# In[45]:


results


# # Saving & Loading Models

# In[51]:


saver = tf.train.Saver()


# In[56]:


### Saving Models
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run((m, c))
    saver.save(sess, './model/reg.ckpt')


# In[57]:


### Loading Models
with tf.Session() as sess:
    saver.restore(sess, './model/reg.ckpt')
    res_m, res_c = sess.run((m, c))


# # Examples

# ## Regression Example: Estimating Housing Prices

# In[2]:


df = pd.read_csv('data/cal_housing_clean.csv')

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X_tn, X_ts, Y_tn, Y_ts = train_test_split(X, Y, test_size=0.3, random_state=101)

X.columns


# In[3]:


mms = MinMaxScaler()
X_tn_mms = mms.fit_transform(X_tn)
X_ts_mms = mms.transform(X_ts)
X_tn_df = pd.DataFrame(data=X_tn_mms, columns=X_tn.columns, index=X_tn.index)
X_ts_df = pd.DataFrame(data=X_ts_mms, columns=X_ts.columns, index=X_ts.index)

hma = tf.feature_column.numeric_column(key='housingMedianAge')
tr = tf.feature_column.numeric_column(key='totalRooms')
tbr = tf.feature_column.numeric_column(key='totalBedrooms')
popn = tf.feature_column.numeric_column(key='population')
hh = tf.feature_column.numeric_column(key='households')
mi = tf.feature_column.numeric_column(key='medianIncome')

feat_cols = [hma, tr, tbr, popn, hh, mi]

dnnr = tf.estimator.DNNRegressor(hidden_units=[6, 6, 6], feature_columns=feat_cols)

input_fn = tf.estimator.inputs.pandas_input_fn(x=X_tn_df, y=Y_tn, batch_size=10, num_epochs=1000, shuffle=True)
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_ts_df, y=Y_ts, batch_size=10, num_epochs=1000, shuffle=False)
pred_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_ts_df, batch_size=10, shuffle=False)


# In[4]:


dnnr.train(input_fn=input_fn, steps=20000)


# In[5]:


dnnr.evaluate(input_fn=eval_input_fn, steps=1000)


# In[6]:


y_ts_pred = []
for prediction in dnnr.predict(input_fn=pred_input_fn):
    y_ts_pred.append(prediction['predictions'][0])
rmse = (((Y_ts.values - y_ts_pred) ** 2).sum() / len(y_ts_pred)) ** 0.5

print(rmse)
print(mse(Y_ts.values, y_ts_pred) ** 0.5)


# ## Classification Example: Income Bracket Prediction

# In[7]:


df = pd.read_csv('data/census_data.csv')
X_raw = df.iloc[:, :-1]
Y_raw = (df.iloc[:, -1] == df.iloc[:, -1].unique()[1]).replace(True, 1).astype(int)

X_tn, X_ts, Y_tn, Y_ts = train_test_split(X_raw, Y_raw, test_size=0.3, random_state=101)
df.columns


# In[9]:


# Numeric columns
age = tf.feature_column.numeric_column(key='age')
education_num = tf.feature_column.numeric_column(key='education_num')
capital_gain = tf.feature_column.numeric_column(key='capital_gain')
capital_loss = tf.feature_column.numeric_column(key='capital_loss')
hours_per_week = tf.feature_column.numeric_column(key='hours_per_week')

# Categorical columns
workclass = tf.feature_column.categorical_column_with_hash_bucket(key='workclass', hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket(key='education', hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket(key='marital_status', hash_bucket_size=1000)
occupation = tf.feature_column.categorical_column_with_hash_bucket(key='occupation', hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket(key='relationship', hash_bucket_size=1000)
race = tf.feature_column.categorical_column_with_hash_bucket(key='race', hash_bucket_size=1000)
gender = tf.feature_column.categorical_column_with_hash_bucket(key='gender', hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket(key='native_country', hash_bucket_size=1000)

# Convert Categorical columns to Embedded
workclass_embedded = tf.feature_column.embedding_column(categorical_column=workclass, dimension=df['workclass'].unique().shape[0])
education_embedded = tf.feature_column.embedding_column(categorical_column=education, dimension=df['education'].unique().shape[0])
marital_status_embedded = tf.feature_column.embedding_column(categorical_column=marital_status, dimension=df['marital_status'].unique().shape[0])
occupation_embedded = tf.feature_column.embedding_column(categorical_column=occupation, dimension=df['occupation'].unique().shape[0])
relationship_embedded = tf.feature_column.embedding_column(categorical_column=relationship, dimension=df['relationship'].unique().shape[0])
race_embedded = tf.feature_column.embedding_column(categorical_column=race, dimension=df['race'].unique().shape[0])
gender_embedded = tf.feature_column.embedding_column(categorical_column=gender, dimension=df['gender'].unique().shape[0])
native_country_embedded = tf.feature_column.embedding_column(categorical_column=native_country, dimension=df['native_country'].unique().shape[0])

# Feature columns
feature_columns = [
    age, 
    education_num, 
    capital_gain, 
    capital_loss, 
    hours_per_week, 
    workclass_embedded, 
    education_embedded,
    marital_status_embedded,
    occupation_embedded,
    relationship_embedded,
    race_embedded,
    gender_embedded,
    native_country_embedded
]

# input functions
input_fn = tf.estimator.inputs.pandas_input_fn(x=X_tn, y=Y_tn, batch_size=10, num_epochs=1000, shuffle=True)
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_ts, y=Y_ts, batch_size=10, num_epochs=1000, shuffle=False)
predict_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_ts, y=Y_ts, batch_size=10, shuffle=False)

# estimator
dnnc = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feature_columns)


# In[10]:


dnnc.train(input_fn=input_fn, steps=5000)


# In[11]:


dnnc.evaluate(input_fn=eval_input_fn, steps=1000)


# In[12]:


predictions = []
for prediction in dnnc.predict(input_fn=predict_input_fn):
    predictions.append(prediction['class_ids'][0])
    
print(classification_report(Y_ts.values, predictions))
