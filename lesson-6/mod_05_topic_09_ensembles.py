import time
import pandas as pd
from sklearn.ensemble import (
    StackingClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# %%

data = pd.read_csv('../datasets/mod_05_topic_09_employee_data.csv')
data.head()

# %%

data.info()

# %%

data['Years'] = data['JoiningYear'].max() - data['JoiningYear']

# %%

data = data.drop(columns='JoiningYear')

# %%

data['PaymentTier'] = data['PaymentTier'].astype(str)

# %%

X_train, X_test, y_train, y_test = (
    train_test_split(
        data.drop('LeaveOrNot', axis=1),
        data['LeaveOrNot'],
        test_size=0.33,
        random_state=42))

# %%
#
df_1 = X_train.copy()
enc_1 = ce.TargetEncoder()
cols = ['Education', 'Years', 'City', 'PaymentTier', 'Age']

# %%
df_1[cols] = enc_1.fit_transform(df_1[cols], y_train)

# %%

df_2 = X_train.copy()
cols = ['Education', 'Years', 'City', 'PaymentTier', 'Age']
for col in cols:
    enc_2 = ce.TargetEncoder()
    df_2[col] = enc_2.fit_transform(df_2[col], y_train)

# %%

encoder = ce.TargetEncoder()

X_train = encoder.fit_transform(X_train, y_train)
X_test = encoder.transform(X_test)

# %%

scaler = StandardScaler().set_output(transform='pandas')

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%

y_train.value_counts(normalize=True)

# %%

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# %%

f1_scores = {}


def measure_f1_time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        predictions = func(*args, **kwargs)
        end_time = time.time()
        f1 = f1_score(args[-1], predictions)
        model_name = args[0].__class__.__name__
        execution_time = end_time - start_time
        f1_scores[model_name] = [f1, execution_time]
        print(f'{model_name} F1 Metric: {f1:.4f}')
        print(f'{model_name} Inference: {execution_time:.4f} s')
        return predictions
    return wrapper


@measure_f1_time_decorator
def predict_with_measure(model, Xt, yt):
    return model.predict(Xt)


# %%

mod_log_reg = LogisticRegression(n_jobs=-1).fit(X_res, y_res)

prd_log_reg = predict_with_measure(mod_log_reg, X_test, y_test)

# %%

mod_rnd_frs = (RandomForestClassifier(
    random_state=42,
    # n_jobs=-1
)
    .fit(X_res, y_res))

prd_rnd_frs = predict_with_measure(mod_rnd_frs, X_test, y_test)

# %%

mod_bag_knn = (BaggingClassifier(
    KNeighborsClassifier(),
    max_samples=0.75,
    max_features=0.75,
    # n_jobs=-1,
    random_state=42)
    .fit(X_res, y_res))

prd_bag_knn = predict_with_measure(mod_bag_knn, X_test, y_test)

# %%

mod_ada_bst = (AdaBoostClassifier(
    algorithm='SAMME',
    random_state=42)
    .fit(X_res, y_res))

prd_ada_bst = predict_with_measure(mod_ada_bst, X_test, y_test)

# %%

mod_grd_bst = (GradientBoostingClassifier(
    learning_rate=0.3,
    subsample=0.75,
    max_features='sqrt',
    random_state=42)
    .fit(X_res, y_res))

prd_grd_bst = predict_with_measure(mod_grd_bst, X_test, y_test)

# %%

clf1 = LogisticRegression()
clf2 = KNeighborsClassifier()
clf3 = GaussianNB()

estimators = [('lnr', clf1),
              ('knn', clf2),
              ('gnb', clf3)]

mod_vot_clf = VotingClassifier(
    estimators=estimators,
    voting='soft').fit(X_res, y_res)

prd_vot_clf = predict_with_measure(mod_vot_clf, X_test, y_test)

# %%

final_estimator = GradientBoostingClassifier(
    subsample=0.75,
    max_features='sqrt',
    random_state=42)

mod_stk_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator).fit(X_res, y_res)

prd_stk_clf = predict_with_measure(mod_stk_clf, X_test, y_test)

# %%

scores = pd.DataFrame.from_dict(
    f1_scores,
    orient='index',
    columns=['f1', 'time'])

scores.sort_values('f1', ascending=False)

# %%
#########
feat_imp_regressor = GradientBoostingClassifier(random_state=42)
X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)
feat_imp_regressor.fit(X, y)
feat_imp = feat_imp_regressor.feature_importances_
feat_imp_df = pd.DataFrame(feat_imp, index=X.columns, columns=['Feature Importance'])

# %%
########
feat_imp_df['Importance Rank'] = feat_imp_df['Feature Importance'].rank(pct=True)

# %%
####
sns.catplot(
    data=feat_imp_df.sort_values(by='Importance Rank', ascending=False).reset_index(),
    y='index',
    x='Importance Rank',
    kind='bar')

plt.show()