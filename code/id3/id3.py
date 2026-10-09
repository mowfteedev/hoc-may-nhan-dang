import math
import pandas as pd

# ---------- Dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()
target, attrs = df.columns[-1], list(df.columns[:-1])
values = {a: df[a].unique() for a in attrs}


# ---------- 1. Entropy ----------
def entropy(D):
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p)


# ---------- 2. Gain ----------
def gain(D, a):
    return entropy(D) - sum(len(sub) / len(D) * entropy(sub) for _, sub in D.groupby(a))


# ---------- 3. Quyết định (ID3) ----------
def id3(D, attrs):
    if D[target].nunique() == 1:
        return D[target].iloc[0]
    if not attrs:
        return D[target].mode()[0]

    best = max(attrs, key=lambda a: gain(D, a))
    print(f"Chọn {best} (Gain = {gain(D, best):.4f})")
    rest = [a for a in attrs if a != best]

    return {best: {v: id3(sub, rest) if len(sub) else D[target].mode()[0]
                   for v in values[best] for sub in [D[D[best] == v]]}}


# ---------- In cây theo chuẩn tree thư mục ----------
def show(tree, prefix=""):
    a = next(iter(tree))
    if not prefix:
        print(a)
    items = list(tree[a].items())
    for i, (v, sub) in enumerate(items):
        is_last = (i == len(items) - 1)
        branch = "└── " if is_last else "├── "
        next_prefix = prefix + ("    " if is_last else "│   ")
        if isinstance(sub, dict):
            print(f"{prefix}{branch}{v} ──▶ {next(iter(sub))}")
            show(sub, next_prefix)
        else:
            print(f"{prefix}{branch}{v} ──▶ {sub}")


def predict(tree, D):
    if not isinstance(tree, dict):
        return tree
    a = next(iter(tree))
    return predict(tree[a].get(D[a], df[target].mode()[0]), D)


# ---------- Chạy ----------
tree = id3(df, attrs)

print("\n===== SƠ ĐỒ CÂY =====")
show(tree)

print("\n===== KẾT QUẢ ĐÁNH GIÁ & DỰ ĐOÁN =====")
# Độ chính xác trên tập dữ liệu huấn luyện
pred = df.apply(lambda r: predict(tree, r), axis=1)
accuracy = (pred == df[target]).mean()
correct = (pred == df[target]).sum()
print(f"Độ chính xác trên tập huấn luyện: {accuracy:.2%} ({correct}/{len(df)} mẫu đúng)")

# Dự đoán cho mẫu dữ liệu mới
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
ket_qua = predict(tree, D_mau)
print(f"Mẫu quan sát mới D: {D_mau}")
print(f"Quyết định dự đoán nhãn cho mẫu D: {ket_qua}")
