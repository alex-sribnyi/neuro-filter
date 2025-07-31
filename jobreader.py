import joblib

model = joblib.load("sgd_model.joblib")

w = model.coef_[0]
b = model.intercept_[0]

print("w =", w)
print("b =", b)
