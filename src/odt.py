import os
import math
import pandas as pd

# 1. Đọc dữ liệu
DATA_FILE = "data/data.csv" if os.path.exists("data/data.csv") else os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
df = pd.read_csv(DATA_FILE)
target = df.columns[-1]
attrs = list(df.columns[:-1])

D_test = {"Tiet_troi": "Nang", "Nhiet_do": "Lanh", "Do_am": "Cao", "Gio": "Manh"}


# 2. Các hàm cốt lõi
def entropy(D):
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p if x > 0)


def gain(D, a):
    w = D[a].value_counts(normalize=True)
    e = 0
    for v, sub in D.groupby(a):
        e += w[v] * entropy(sub)
    return entropy(D) - e


def odt(D, attrs):
    return sorted(attrs, key=lambda a: gain(D, a), reverse=True)



