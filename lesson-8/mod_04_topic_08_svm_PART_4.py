import warnings

import joblib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, TargetEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from custom_transformer import LenOfDescriptionTransformer, OtherCustomTransformers, kbins_transformer
    
pd.set_option("future.no_silent_downcasting", True)

# %%

try:
    data = pd.read_csv('./datasets/mod_04_topic_08_petfinder_data.csv.gz')
except FileNotFoundError:
    data = pd.read_csv('../datasets/mod_04_topic_08_petfinder_data.csv.gz')
    
data.info()

# %%

data['AdoptionSpeed'] = np.where(data['AdoptionSpeed'] == 4, 0, 1)

# %%

X_train, X_test, y_train, y_test = (
    train_test_split(
        data.drop('AdoptionSpeed', axis=1),
        data['AdoptionSpeed'],
        test_size=0.2,
        random_state=42))

# %% 

cat_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        #('encoder', ce.TargetEncoder()) 
    ])

num_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('kbins', KBinsDiscretizer(encode='ordinal')),
        ('kbins_str', FunctionTransformer(kbins_transformer)) 
    ])

# %%

col_processor = (ColumnTransformer(
    transformers=[
        ('cat',
         cat_transformer,
         make_column_selector(dtype_include=object)),
        ('num',
         num_transformer,
         make_column_selector(dtype_exclude=object))],
    n_jobs=-1,
    verbose_feature_names_out=False)
    .set_output(transform='pandas'))

# %%

clf_estimator = RandomForestClassifier(n_jobs=-1, random_state=42)

# %%

lod_transformer = LenOfDescriptionTransformer()
other_transformer = OtherCustomTransformers()

clf_pipe_model = (Pipeline(
    steps=[
        ('lod_transformer', lod_transformer),
        ('other_transformers', other_transformer),
        ('col_processor', col_processor),
        ('encoder', TargetEncoder().set_output(transform='pandas')),
        ('scaler', StandardScaler().set_output(transform='pandas')),
        ('clf_estimator', clf_estimator)
    ]))


clf_model = clf_pipe_model.fit(X_train, y_train)

# %%

pred_pipe = clf_model.predict(X_test)

# print(confusion_matrix(y_test, pred_pipe))
print(f"Pipe's accuracy is: {accuracy_score(y_test, pred_pipe):.1%}")

# %%

pet = pd.DataFrame(
    data={
        'Type': 'Cat', # Dolphin
        'Age': 3,
        'Breed1': 'Tabby',
        'Gender': 'Male',
        'Color1': 'Black',
        'Color2': 'White',
        'MaturitySize': 'Small',
        'FurLength': 'Short',
        'Vaccinated': 'No',
        'Sterilized': 'No',
        'Health': 'Healthy',
        'Fee': 5,
        'Description': f'{"a"*1500}',
        'PhotoAmt': 2
    },
    index=[0])

pet_pred_proba = clf_model.predict_proba(pet).flatten()

print(f'This pet has a {pet_pred_proba[1]:.1%} probability "of getting adopted"')
print('==========================')

# %%

X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)

cv_results = cross_val_score(
    estimator=clf_pipe_model,
    X=X,
    y=y,
    scoring='accuracy',
    cv=5,
    verbose=1)

acc_cv = cv_results.mean()

print(f"Pipe's accuracy on CV: {acc_cv:.1%}")

# %%

parameters = {
    'clf_estimator__max_depth': (None, 5),
    'clf_estimator__max_features': ('sqrt', 0.75)}

search = (GridSearchCV(
    estimator=clf_pipe_model,
    param_grid=parameters,
    scoring='accuracy',
    cv=5,
    refit=False)
    .fit(X, y))

# %%

parameters_best = search.best_params_
clf_pipe_model = clf_pipe_model.set_params(**parameters_best)

model_upd = clf_pipe_model.fit(X, y)

# %%

cv_results_upd = cross_val_score(
    estimator=model_upd,
    X=X,
    y=y,
    scoring='accuracy',
    cv=5)

acc_cv_upd = cv_results_upd.mean()
print(f"Pipe's UPDATED accuracy on CV: {acc_cv_upd:.1%}")

# %%

with open('final_pipeline_with_model.pkl', 'wb') as f:
    joblib.dump(model_upd, f)
