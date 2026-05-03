import joblib
import numpy as np

p = joblib.load('models/preprocessor.pkl')
m = joblib.load('models/best_model.pkl')

test = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
processed = p.transform(test)
pred = m.predict(processed)
prob = m.predict_proba(processed)[0][1]

print("Prediction:", pred)
print("Probability:", prob)
print("✅ Model works!")