import math
import os
import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()   # Đọc bảng và loại bỏ dòng thiếu dữ liệu
# df = df.drop(columns=["ID"])          # Bỏ cột ID/STT nếu có

target = df.columns[-1]                 # Cột cuối cùng là nhãn cần dự đoán (Class/Label)
values = {a: df[a].unique() for a in df.columns[:-1]}   # Mọi giá trị có thể của từng thuộc tính (tránh khuyết nhánh)


# ---------- Entropy ----------
def entropy(d):
    p = d[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p)


# ---------- Information Gain ----------
def gain(d, attr):
    g = entropy(d)
    for v, sub in d.groupby(attr):
        g -= len(sub) / len(d) * entropy(sub)
    return g


# ---------- Thuật toán ID3 ----------
def id3(d, attrs):
    major = d[target].mode()[0]                       # nhãn đa số của tập hiện tại

    attrs = [a for a in attrs if d[a].nunique() > 1]  # chỉ giữ thuộc tính còn chia được

    if d[target].nunique() == 1 or not attrs:         # thuần nhất hoặc hết thuộc tính
        return major

    gains = {a: gain(d, a) for a in attrs}            # tính Gain một lần cho mỗi thuộc tính
    best = max(gains, key=gains.get)
    print(f"Chọn {best} (Gain = {gains[best]:.4f})")

    rest = [a for a in attrs if a != best]
    children = {}
    for v in values[best]:                            # tạo đủ nhánh cho mọi giá trị
        sub = d[d[best] == v]
        children[v] = id3(sub, rest) if len(sub) else major   # nhánh rỗng -> nhãn đa số của nút cha
    return {"attr": best, "major": major, "children": children}


# ---------- In cây ----------
def show(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "=> " + str(tree))
        return
    for v, sub in tree["children"].items():
        print(f"{indent}{tree['attr']} = {v}")
        show(sub, indent + "    ")


# ---------- Dự đoán ----------
def predict(tree, x):
    while isinstance(tree, dict):
        # giá trị chưa từng thấy -> trả về nhãn đa số tại nút đó
        tree = tree["children"].get(x[tree["attr"]], tree["major"])
    return tree


# ---------- Chạy ----------
tree = id3(df, list(df.columns[:-1]))

print("\n===== CÂY QUYẾT ĐỊNH =====")
show(tree)

# Độ chính xác trên tập huấn luyện
pred = df.apply(lambda r: predict(tree, r), axis=1)
print(f"\nĐộ chính xác (train): {(pred == df[target]).mean():.2%}")

# Dự đoán mẫu mới (sửa theo tên cột/giá trị trong file của bạn)
# mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
# print("Quyết định:", predict(tree, mau))
