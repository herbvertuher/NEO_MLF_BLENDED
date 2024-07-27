import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

# %%

data = pd.read_csv('../datasets/mod_03_topic_06_diabets_data.csv')
#data.head()

# %%
np.random.seed(999)
data['Hair_Color'] = np.random.choice(['Black', 'Brown'], data.shape[0])

# %%

data.info()

# %%

X, y = (data.drop('Outcome', axis=1), data['Outcome'])

# %%

ax = sns.scatterplot(x=X['Glucose'], y=X['BMI'], hue=y)
ax.vlines(x=[120, 160],
          ymin=0,
          ymax=X['BMI'].max(),
          color='black',
          linewidth=0.75)

plt.show()

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42)

# %%

cats = X_train.select_dtypes('object').columns

# , handle_unknown='ignore'
enc = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

X_train_enc = enc.fit_transform(X_train[cats])
X_test_enc = enc.transform(X_test[cats])

X_train = pd.concat([X_train, X_train_enc], axis=1).drop(columns=cats)
X_test = pd.concat([X_test, X_test_enc], axis=1).drop(columns=cats)
#
# %%

clf = (tree.DecisionTreeClassifier(
    random_state=42)
    .fit(X_train, y_train))

y_pred = clf.predict(X_test)

acc = balanced_accuracy_score(y_test, y_pred)

print(f'Acc.: {acc:.1%}')

# %%

plt.figure(figsize=(80, 15), dpi=196)

tree.plot_tree(clf,
               feature_names=X_train.columns,
               filled=True,
               fontsize=6,
               class_names=list(map(str, y_train.unique())),
               # proportion=True,
               # precision=2,
               rounded=True)

#plt.savefig('./decision_tree_plot.png')
plt.show()

# %%

y_train.value_counts(normalize=True)

# %%

sm = SMOTE(random_state=42, k_neighbors=10)
X_res, y_res = sm.fit_resample(X_train, y_train)

y_res.value_counts(normalize=True)

# %%

clf_upd = (tree.DecisionTreeClassifier(
    max_depth=4,
    random_state=42)
    .fit(X_res, y_res))

y_pred_upd = clf_upd.predict(X_test)

acc = balanced_accuracy_score(y_test, y_pred_upd)

print(f'Acc.: {acc:.1%}')

# %%

plt.figure(figsize=(25, 7))

tree.plot_tree(clf_upd,
               feature_names=X_train.columns,
               filled=True,
               fontsize=8,
               class_names=list(map(str, y_res.unique())),
               # proportion=True,
               # precision=2,
               rounded=True)

plt.show()

# %%

(pd.Series(
    data=clf_upd.feature_importances_,
    index=X_train.columns)
    .sort_values(ascending=True)
    .plot
    .barh())

plt.show()

# %%

two_samples = data.iloc[:2]
X_new = two_samples.drop(columns='Outcome')
y_new = two_samples['Outcome']

X_new['Hair_Color'] = 'Blonde'

encoded_cats = enc.transform(X_new[cats])
X_new = pd.concat([X_new, encoded_cats], axis=1)
X_new = X_new.drop(columns=cats)

# %%

new_predict = clf_upd.predict(X_new)

confusion_matrix(y_new, new_predict)

# %%

