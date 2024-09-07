import numpy as np
import pandas as pd
import seaborn as sns

# %%
# step 1

df1 = pd.read_csv('./datasets/peoples.csv', sep=',')

# %%
# step 2

df2 = pd.read_csv('./datasets/peoples.csv', index_col='ID')

# df2.loc[210]
df1[df1['ID'] == 210]

# %%
# step 3

df3 = df2.dropna()

# %%
# step 4

df4 = df2.copy()
cols_to_drop = ['Name', 'Address', 'Phone']
df4 = df4.drop(cols_to_drop, axis=1)
# df4 = df4.drop(columns=cols_to_drop)

# %%
# step 5

nan_mask = df4.isnull().any(axis=1)

df5 = df4[nan_mask]

# %%
# step 6

# df6 = df4.drop(index=3)
# df6 = df4[~nan_mask]
df6 = df4.dropna()

# df_ff = df2[nan_mask]
# df_ff = df2[cols_to_drop]
df_ff = df2.loc[nan_mask, cols_to_drop]

# %%
# step 7

test_df = pd.DataFrame({'one': [1, 2, 3], 
                        'two': ['a', 'b', 'c']
                        })

# %%

new_df = test_df

# %%

new_df.iloc[0, 1] = 8
new_df.loc[0, 'two'] = 88
new_df['two'][0] = 888

# %%
# step 8

duplicates_mask = df6.duplicated()

# df7 = df6[~duplicates_mask]
df7 = df6.drop_duplicates()


# %%
# step 9
sns.heatmap(df7.corr(numeric_only=True), cmap='coolwarm', annot=True)


# %%
# step 10
df7.dtypes

# %%
# step 11

# df7.groupby('Country')['Salary'].mean()

# df7.groupby('Pet')['Salary'].mean()

# df7.groupby('Qualification')['Salary'].mean()

# df7.groupby(by=['Country', 'Qualification'])['Salary'].mean()
df7.groupby(by=['Country', 'Qualification'])['Salary'].apply(np.mean)

# %%
# step 12

df7['Country'].value_counts(normalize=True)

# %%
# step 13

df8 = df7.drop(['Country', 'Pet', 'Qualification'], axis=1)

# %%
# step hist

m = df8.melt()

g = sns.FacetGrid(m,
                  col='variable',
                  col_wrap=3,
                  sharex=False,
                  sharey=False)

g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()

# %%
# step 14

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# %%
# step 15

X_train, X_test, y_train, y_test = train_test_split(
                                            df8.drop('Salary', axis=1),
                                            df8['Salary'],
                                            test_size=0.2,
                                            random_state=42)

# %%
# step 16

scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_trn = StandardScaler().set_output(transform='pandas').fit(X_train)
scaler_all = StandardScaler().set_output(transform='pandas').fit(df8.drop('Salary', axis=1))

# %%
# step 17

model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


# %%
# step 18

r_sq = model.score(X_test_scaled, y_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'R2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2%}')


# %%

from sklearn.datasets import fetch_california_housing


california_housing = fetch_california_housing(as_frame=True)

df_data = california_housing.data
df_data1 = california_housing['data']

df_target = california_housing.target

df_frame = california_housing.frame


