from sklearn.base import BaseEstimator, TransformerMixin


class LenOfDescriptionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X['LenOfDescription'] = X['Description'].str.len()
        X['LenOfDescription'] = X['LenOfDescription'].fillna(0).astype(int)
        
        return X
        

class OtherCustomTransformers(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X['Fee'] = X['Fee'].astype(bool).astype(int).astype(str)
        X = X.drop('Description', axis=1)

        return X
        
        