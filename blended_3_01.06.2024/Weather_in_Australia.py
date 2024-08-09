# Blended 3

import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree

# %%

data = pd.read_csv('../datasets/mod_03_topic_05_weather_data.csv.gz')
print(data.shape)

# %%

print(data.dtypes)

# %%

print(data.isna().mean().sort_values(ascending=False))

# %%

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    tmp = (data
           .groupby('Location')
           .apply(lambda x:
                  x.drop(['Location', 'Date'], axis=1)
                  .isna()
                  .mean()))

plt.figure(figsize=(9, 15), dpi=100)

ax = sns.heatmap(tmp,
                 cmap='Blues',
                 linewidth=0.5,
                 square=True,
                 cbar_kws=dict(
                     location="bottom",
                     pad=0.01,
                     shrink=0.25))

ax.xaxis.tick_top()
ax.tick_params(axis='x', labelrotation=90)

plt.show()

# %%

data = data[data.columns[data.isna().mean().lt(0.35)]]

data = data.dropna(subset='RainTomorrow')

# %%

data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')

# %%

melted = data_num.melt()

g = sns.FacetGrid(melted,
                  col='variable',
                  col_wrap=4,
                  sharex=False,
                  sharey=False,
                  aspect=1.25)

g.map(sns.histplot, 'value', bins=20)

g.set_titles(col_template='{col_name}')

g.tight_layout()

plt.show()

# %%

print(data_cat.nunique())

# %%

data_cat.apply(lambda x: x.unique()[:5])

# %%

data_cat['Date'] = pd.to_datetime(data_cat['Date'])

data_cat['Year'] = data_cat['Date'].dt.year.astype(str)
data_cat['Month'] = data_cat['Date'].dt.month.astype(str)
    
data_cat = data_cat.drop('Date', axis=1)

data_cat[['Year', 'Month']].head()

# %%

X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test = (
    train_test_split(
        data_num,
        data_cat.drop('RainTomorrow', axis=1),
        data['RainTomorrow'],
        test_size=0.3,
        random_state=99))

# %%

num_imputer = SimpleImputer(strategy='mean').set_output(transform='pandas')
X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test_num)


# %%

cat_imputer = SimpleImputer(
    strategy='most_frequent').set_output(transform='pandas')
X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.transform(X_test_cat)


# %%

scaler = StandardScaler().set_output(transform='pandas')

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# %%

encoder = (OneHotEncoder(drop='if_binary',
                          sparse_output=False)
            .set_output(transform='pandas'))

X_train_cat = encoder.fit_transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)

X_train_cat.shape

# %%

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

X_train.shape

# %%

print(y_train.value_counts(normalize=True))

# %%

clf = (LogisticRegression(random_state=99, 
                          class_weight='balanced')
       .fit(X_train, y_train))

pred = clf.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_test, pred)

plt.show()

print(classification_report(y_test, pred))

# %%

# Pred proba LR

threshold = 0.5

y_pred_proba = pd.Series(clf.predict_proba(X_test)[:,1])
y_pred = y_pred_proba.apply(lambda x: 'Yes' if x > threshold else 'No')

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.show()

print(classification_report(y_test, y_pred))

df_proba = pd.DataFrame()
df_proba['y_pred_proba'] = y_pred_proba
df_proba['y_test'] = y_test

target_class = 'Yes'

df_proba[
    (df_proba['y_pred_proba'] < threshold) &
    (df_proba['y_test'] == target_class)
]['y_pred_proba'].hist(bins=20, color='red');

df_proba[
    (df_proba['y_pred_proba'] >= threshold) &
    (df_proba['y_test'] == target_class)
]['y_pred_proba'].hist(bins=20, color='green');

plt.title(f'class = {target_class}');


# %%
# %%
# %%
# %%
# %%
# %%

# Forest

X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test = (
    train_test_split(
        data_num,
        data_cat.drop('RainTomorrow', axis=1),
        data['RainTomorrow'],
        test_size=0.2,
        random_state=99))

# %%

num_imputer = SimpleImputer().set_output(transform='pandas')
X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test_num)


# %%

cat_imputer = SimpleImputer(
    strategy='most_frequent').set_output(transform='pandas')
X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.transform(X_test_cat)

# %%

encoder = (OneHotEncoder(drop='if_binary',
                          sparse_output=False)
            .set_output(transform='pandas'))

X_train_cat = encoder.fit_transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)

X_train_cat.shape

# %%

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

# %%

dtc = (tree.DecisionTreeClassifier(random_state=99, 
                          class_weight='balanced',
                          max_depth=5)
       .fit(X_train, y_train))

pred = dtc.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_test, pred)

plt.show()

print(classification_report(y_test, pred))

# %% 

# Pred proba DTC

threshold = 0.6

y_pred_proba = pd.Series(dtc.predict_proba(X_test)[:,1])
y_pred = y_pred_proba.apply(lambda x: 'Yes' if x > threshold else 'No')

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.show()

print(classification_report(y_test, y_pred))

df_proba = pd.DataFrame()
df_proba['y_pred_proba'] = y_pred_proba
df_proba['y_test'] = y_test

target_class = 'Yes'

df_proba[
    (df_proba['y_pred_proba'] < threshold) &
    (df_proba['y_test'] == target_class)
]['y_pred_proba'].hist(bins=20, color='red');

df_proba[
    (df_proba['y_pred_proba'] >= threshold) &
    (df_proba['y_test'] == target_class)
]['y_pred_proba'].hist(bins=15, color='green');

plt.title(f'class = {target_class}');