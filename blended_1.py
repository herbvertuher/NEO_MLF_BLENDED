import numpy as np
import pandas as pd
import seaborn as sns

# %% 1

df1 = pd.read_csv('./datasets/peoples.csv', sep=',')

# %% 2

df2 = pd.read_csv('./datasets/peoples.csv', index_col='ID')

df2.loc[210]
#df2[df2['ID'] == 210]

# %% 3

df3 = df2.dropna()

# %% 4

df4 = df2.copy()
cols_to_drop = ['Name', 'Address', 'Phone']
df4 = df4.drop(cols_to_drop, axis=1)

# %% 5

nan_mask = df4.isnull().any(axis=1)

df5 = df4[nan_mask]

# %% 6

df6 = df4.drop(index=3)
df6 = df4[~nan_mask]
df6 = df4.dropna()

# %% 7

test_df = pd.DataFrame({'one': [1, 2, 3], 
                        'two': ['a', 'b', 'c']
                        })

# %% -

new_df = test_df

# %% -

new_df.iloc[0, 1] = 8
new_df.loc[0, 'two'] = 88
new_df['two'][0] = 888

# %% 8

duplicates_mask = df6.duplicated()

df7 = df6[~duplicates_mask]
#df7 = df6.drop_duplicates()


# %% 9
sns.heatmap(df7.corr(numeric_only=True), cmap='coolwarm', annot=True)


# %% 10
df7.dtypes

# %% 11

#df7.groupby('Country')['Salary'].mean()

#df7.groupby('Pet')['Salary'].mean()

#df7.groupby('Qualification')['Salary'].mean()

#df7.groupby(by=['Country', 'Qualification'])['Salary'].mean()
#df7.groupby(by=['Country', 'Qualification'])['Salary'].apply(np.mean)

# %% 12

df7['Country'].value_counts(normalize=True)

# %% 13

df8 = df7.drop(['Country', 'Pet', 'Qualification'], axis=1)

# %% hist

m = df8.melt()

g = sns.FacetGrid(m,
                  col='variable',
                  col_wrap=3,
                  sharex=False,
                  sharey=False)

g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()

# %% 14

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# %% 15

X_train, X_test, y_train, y_test = train_test_split(
                                            df8.drop('Salary', axis=1),
                                            df8['Salary'],
                                            test_size=0.2,
                                            random_state=42)

# %% 16

scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% 17

base_model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = base_model.predict(X_test_scaled)


# %% 18

r_sq = base_model.score(X_train_scaled, y_train)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'R2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')

# %% 19
