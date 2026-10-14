import os
import math
import pandas as pd

DATA_FILE = "data/data.csv" if os.path.exists("data/data.csv") else os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
df = pd.read_csv(DATA_FILE)
target = df.columns[-1]
attrs = list(df.columns[:-1])

D_test = {"Tiet_troi": "Nang", "Nhiet_do": "Lanh", "Do_am": "Cao", "Gio": "Manh"}


def entropy(D):
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p if x > 0)


def gain(D, a):
    return entropy(D) - sum(len(sub) / len(D) * entropy(sub) for _, sub in D.groupby(a))


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


tree = id3(df, attrs)
print("Completed!\n" + "-" * 35)

H_D = entropy(df)
counts = df[target].value_counts().to_dict()
counts_str = ", ".join(f"{cnt} {lbl}" for lbl, cnt in counts.items())
print(f"Entropy tổng H(D) = {H_D:.4f} ({counts_str})")
print("-" * 35)

for a in attrs:
    print(f"Gain(D, {a}) = {gain(df, a):.4f}")
best_root = max(attrs, key=lambda a: gain(df, a))
print(f"=> Nút gốc (Gain max): {best_root} = {gain(df, best_root):.4f}")
print("-" * 35)

show(tree)
print("-" * 35)

D = D_test
print(f"D = {D}")
print(f"=> Quyết định ID3: {predict(tree, D)}")
