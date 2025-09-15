import pandas as pd
from cindex.model import fit_logistic, predict_proba

def test_fit_predict():
    df = pd.DataFrame({
        "S":[-5.7,-5.1,-5.9,-4.9],
        "q6":[0.42,0.20,0.50,0.18],
        "p2":[0.65,0.30,0.70,0.25],
        "y":[1,0,1,0],
    })
    pack = fit_logistic(df)
    proba = predict_proba(df, pack)
    assert proba.min() >= 0 and proba.max() <= 1
