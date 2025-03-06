from sklearn.utils import all_estimators
from onnxconverter_common.container import CommonSklearnModelContainer

for e in all_estimators():
    print(e[0])