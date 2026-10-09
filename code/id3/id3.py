import math
import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()
target = df.columns[-1]
attrs = list(df.columns[:-1])


# ---------- 1. Các hàm cơ bản ----------
def entropy(D):
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p if x > 0)


def gain(D, a):
    return entropy(D) - sum(len(sub) / len(D) * entropy(sub) for _, sub in D.groupby(a))


# ---------- 2. Quyết định (ID3) ----------
def id3(D, attrs):
    if D[target].nunique() == 1:
        return D[target].iloc[0]
    if not attrs:
        return D[target].mode()[0]

    best = max(attrs, key=lambda a: gain(D, a))
    rest = [a for a in attrs if a != best]

    tree = {}
    for v in df[best].unique():
        sub = D[D[best] == v]
        tree[v] = id3(sub, rest) if len(sub) else D[target].mode()[0]
    return {best: tree}


def show(tree, indent=""):
    a = next(iter(tree))
    for v, sub in tree[a].items():
        if isinstance(sub, dict):
            print(f"{indent}[{a}] = {v}:")
            show(sub, indent + "    ")
        else:
            print(f"{indent}[{a}] = {v}  ──▶  {sub}")


def predict(tree, D):
    if not isinstance(tree, dict):
        return tree
    a = next(iter(tree))
    return predict(tree[a].get(D[a], df[target].mode()[0]), D)


# ---------- Chạy thử nghiệm ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"D = {D_mau}")

tree = id3(df, attrs)
print("Completed!\n" + "-" * 35)
show(tree)
print("-" * 35)
print(f"=> Quyết định ID3: {predict(tree, D_mau)}")
