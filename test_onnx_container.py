import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxconverter_common.container import CommonSklearnModelContainer
import onnxruntime as rt

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

container = CommonSklearnModelContainer(model)

initial_types = [('float_input', FloatTensorType([None, 4]))]

onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=12)

with open("rf_iris.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

sess = rt.InferenceSession("rf_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onnx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]

pred_sklearn = model.predict(X_test)
print(f"ONNX 预测结果： {pred_onnx}")
print(f"scikit-learn 预测结果： {pred_sklearn}")
print(f"预测结果一致: {np.array_equal(pred_onnx, pred_sklearn)}")