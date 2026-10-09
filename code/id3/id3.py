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
    return -sum(x * math.log2(x) for x in p if x > 0)


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
    rest = [a for a in attrs if a != best]

    return {best: {v: id3(sub, rest) if len(sub) else D[target].mode()[0]
                   for v in values[best] for sub in [D[D[best] == v]]}}


# ---------- In sơ đồ cây ----------
def show(tree, indent=""):
    a = next(iter(tree))
    for v, sub in tree[a].items():
        if isinstance(sub, dict):
            print(f"{indent}[{a}] = {v}:")
            show(sub, indent + "    ")
        else:
            print(f"{indent}[{a}] = {v}  ──▶  {sub}")


# ---------- Dự đoán ----------
def predict(tree, D):
    if not isinstance(tree, dict):
        return tree
    a = next(iter(tree))
    return predict(tree[a].get(D[a], df[target].mode()[0]), D)


# ==============================================================
# THỰC THI THEO ĐÚNG 4 BƯỚC YÊU CẦU
# ==============================================================

# 1. Entropy tổng
print("===== 1. ENTROPY TỔNG =====")
H_tong = entropy(df)
counts = df[target].value_counts().to_dict()
counts_str = ", ".join(f"{cnt} {lbl}" for lbl, cnt in counts.items())
print(f"Entropy tổng H(D): {H_tong:.4f} ({counts_str})")

# 2. Tính Gain và lấy cái lớn nhất
print("\n===== 2. TÍNH GAIN VÀ LẤY CÁI LỚN NHẤT =====")
for a in attrs:
    print(f"Gain(D, {a:<8}) = {gain(df, a):.4f}")
best_root = max(attrs, key=lambda a: gain(df, a))
print(f"=> Lấy thuộc tính có Gain lớn nhất làm Nút gốc: {best_root} (Gain = {gain(df, best_root):.4f})")

# 3. Vẽ sơ đồ cây
tree = id3(df, attrs)
print("\n===== 3. VẼ SƠ ĐỒ CÂY =====")
show(tree)

# 4. Độ chính xác và mẫu khác
print("\n===== 4. ĐỘ CHÍNH XÁC VÀ MẪU KHÁC =====")
pred = df.apply(lambda r: predict(tree, r), axis=1)
accuracy = (pred == df[target]).mean()
correct = (pred == df[target]).sum()
print(f"Độ chính xác trên tập huấn luyện: {accuracy:.2%} ({correct}/{len(df)} mẫu đúng)")

D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
ket_qua = predict(tree, D_mau)
print(f"Mẫu quan sát khác D: {D_mau}")
print(f"=> Quyết định dự đoán nhãn cho mẫu D: {ket_qua}")
