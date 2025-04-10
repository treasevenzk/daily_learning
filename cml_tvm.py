import os
import sys
import tvm
import pickle
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from decision_compiler.model import build_model

def load_data(data_dir, dataset):
    #data_name = os.path.join(data_dir, dataset + ".dat")
    data_name = "test_datasets/iris_test.dat"
    data = pickle.load(open(data_name, 'rb'))
    data = data.astype(np.float32)
    return data

def load_model(model_dir, func_name, dataset):
    #filename = func_name + "_" + dataset + ".sav"
    filename = "test_models/iris_decision_tree_model.sav"
    #filename = os.path.join(model_dir, filename)
    clf = pickle.load(open(filename, 'rb'))
    return clf

def convert_clf_classes_to_int(clf):
    if(hasattr(clf, "classes_")):
        clf.classes_ = [int(i) for i in clf.classes_]
        clf.classes_ =np.array(clf.classes_)
    return clf

def bench_cmlcompiler(model, number, input_data, check_flag=False):
    """
    Benchmarking cmlcompiler model
    Input model and data
    Return average exection time 
    """

    input_data = tvm.nd.array(input_data)
    start_time = time.perf_counter()
    for i in range(number):
        try:
            load_time, exec_time, store_time, take_time, out_data = model.run(input_data, breakdown=True)
        except Exception as e:
            print(e)
            load_time = exec_time = store_time = take_time = 0
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / number
    if(check_flag == True):
        return load_time, exec_time, store_time, take_time, avg_time, out_data
    else:
        return load_time, exec_time, store_time, take_time, avg_time


def _model(df, breakdown, sklearn_func, dataset, model_dir, data_dir, number, target_list, batch_size, check_flag):
    func_name = str(sklearn_func).split("\'")[1].split(".")[-1]
    data = load_data(data_dir, dataset)
    clf = load_model(model_dir, func_name, dataset)
    clf = convert_clf_classes_to_int(clf)
    results = [func_name, dataset]
    for target in target_list:
        model = build_model(clf, data.shape, batch_size=batch_size, target=target, sparse_replacing=False, dtype_converting=False, elimination=False)
        load_time, exec_time, store_time, take_time, cmlcompiler_time = bench_cmlcompiler(model, number, data, check_flag=check_flag)
        IO_time = load_time + store_time
        computation_time = exec_time + take_time
        results.append(cmlcompiler_time)
        breakdown_results = [func_name, dataset, target, IO_time, computation_time]
        breakdown.loc[len(breakdown)] = breakdown_results
    df.loc[len(df)] = results
    return df, breakdown


def test_framework(models, datasets, target_list, n_repeat, model_dir, data_dir, batch_size, check_flag=False):
    columns = ["model", "dataset"] + target_list
    df = pd.DataFrame(columns=columns)
    breakdown_columns = ["model", "dataset", "target", "IO", "computation"]
    breakdown = pd.DataFrame(columns=breakdown_columns)
    for model in models:
        for dataset in datasets:
            for j in range(n_repeat):
                df, breakdown = _model(df, breakdown, model, dataset, model_dir, data_dir, 1, target_list, batch_size, check_flag)
    return df, breakdown

models = [DecisionTreeClassifier]
datasets = ["year"]
model_dir = "test_models"
data_dir = "test_datasets"

target_list = ["llvm -mcpu=core-avx2"]
n_model = int(sys.argv[1])
savefile = sys.argv[2]
breakdown_file = sys.argv[3]
if(sys.argv[4] == "False"):
    save_header = False
else:
    save_header = True
test_models = [models[n_model]]
n_repeat = 1

df, breakdown = test_framework(test_models, datasets, target_list, n_repeat, model_dir, data_dir, batch_size=None, check_flag=False)
df.to_csv(savefile, mode = "a", index=False, header=save_header)
breakdown.to_csv(breakdown_file, mode = "a", index=False, header=save_header)