import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hummingbird.ml import convert, load

num_classes = 2
X = np.random.rand(100000, 28)
y = np.random.randint(num_classes, size=100000)

skl_model = RandomForestClassifier(n_estimators=10, max_depth=10)
skl_model.fit(X, y)

model = convert(skl_model, 'pytorch')

model.predict(X)

model.to('cuda')
model.predict(X)

model.save('hb_model')

model = load('hb_model', override_flag=True)