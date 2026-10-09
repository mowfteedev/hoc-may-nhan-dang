import math
import pandas as pd

# ---------- Đọc dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()
# df = df.drop(columns=["ID"])          # bỏ cột ID/STT nếu có

target = df.columns[-1]                 # cột cuối là nhãn
attrs = list(df.columns[:-1])           # các thuộc tính
values = {a: df[a].unique() for a in attrs}   # mọi giá trị có thể của từng thuộc tính


# ---------- 1. Tính Entropy ----------
def entropy(D):
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p)


# ---------- 2. Tính Gain ----------
def gain(D, a):
    g = entropy(D)
    for v, sub in D.groupby(a):
        g -= len(sub) / len(D) * entropy(sub)
    return g


# ---------- 3. Đưa ra quyết định (ID3) ----------
def id3(D, attrs):
    major = D[target].mode()[0]                       # nhãn đa số của tập hiện tại

    attrs = [a for a in attrs if D[a].nunique() > 1]  # chỉ giữ thuộc tính còn chia được

    if D[target].nunique() == 1 or not attrs:         # thuần nhất hoặc hết thuộc tính
        return major

    gains = {a: gain(D, a) for a in attrs}            # tính Gain một lần cho mỗi thuộc tính
    best = max(gains, key=gains.get)                  # chọn thuộc tính Gain lớn nhất
    print(f"Chọn {best} (Gain = {gains[best]:.4f})")

    rest = [a for a in attrs if a != best]
    children = {}
    for v in values[best]:                            # tạo đủ nhánh cho mọi giá trị
        sub = D[D[best] == v]
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
def predict(tree, D_mau):
    while isinstance(tree, dict):
        tree = tree["children"].get(D_mau[tree["attr"]], tree["major"])
    return tree


# ---------- Chạy ----------
tree = id3(df, attrs)

print("\n===== CÂY QUYẾT ĐỊNH =====")
show(tree)

# Độ chính xác trên tập huấn luyện
pred = df.apply(lambda r: predict(tree, r), axis=1)
print(f"\nĐộ chính xác (train): {(pred == df[target]).mean():.2%}")

# Dự đoán mẫu mới
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print("Quyết định:", predict(tree, D_mau))
