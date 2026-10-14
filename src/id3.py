# ==============================================================
# TC1: KHAI BÁO THƯ VIỆN SỬ DỤNG CHO THUẬT TOÁN (0.5đ)
# ==============================================================
import io
import math
import pandas as pd

# ==============================================================
# TC2: TẠO DỮ LIỆU TRAIN RIÊNG CHO BÀI TOÁN (1.5đ)
# ==============================================================
train_data = """Ngay,Tiet_troi,Nhiet_do,Do_am,Gio,Choicau
1,Nang,Nong,Cao,Yeu,Khong
2,Nang,Nong,Cao,Manh,Khong
3,Nhieu_may,Nong,Cao,Yeu,Co
4,Mua,Mat,Cao,Yeu,Co
5,Mua,Lanh,Thuong,Yeu,Co
6,Mua,Lanh,Thuong,Manh,Khong
7,Nhieu_may,Lanh,Thuong,Manh,Co
8,Nang,Mat,Cao,Yeu,Khong
9,Nang,Lanh,Thuong,Yeu,Co
10,Mua,Mat,Thuong,Yeu,Co
11,Nang,Mat,Thuong,Manh,Co
12,Nhieu_may,Mat,Cao,Manh,Co
13,Nhieu_may,Nong,Thuong,Yeu,Co
14,Mua,Mat,Cao,Manh,Khong"""

df = pd.read_csv(io.StringIO(train_data))
if "Ngay" in df.columns:
    df = df.drop(columns=["Ngay"])
target = df.columns[-1]
attrs = list(df.columns[:-1])

# ==============================================================
# TC3: TẠO DỮ LIỆU TEST (1.0đ)
# ==============================================================
D_test = {"Tiet_troi": "Nang", "Nhiet_do": "Lanh", "Do_am": "Cao", "Gio": "Manh"}


# ==============================================================
# TC4: VIẾT ĐÚNG CÔNG THỨC SỬ DỤNG CHO THUẬT TOÁN (1.0đ)
# Công thức Entropy: H(D) = - Σ p * log2(p)
# Công thức Gain: Gain(D, a) = H(D) - Σ (|D_v| / |D|) * H(D_v)
# ==============================================================

# ==============================================================
# TC5: TÍNH THAM SỐ THỨ NHẤT CỦA THUẬT TOÁN - ENTROPY H(D) (1.5đ)
# ==============================================================
def entropy(D):
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p if x > 0)


# ==============================================================
# TC6: TÍNH THAM SỐ THỨ HAI CỦA THUẬT TOÁN - GAIN & DỰNG CÂY (1.5đ)
# ==============================================================
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


# ==============================================================
# TC7: NHẬP DỮ LIỆU TEST (1.0đ)
# ==============================================================
D = D_test
print(f"D = {D}")

# ==============================================================
# TC8 & TC9: DỰNG CÂY VÀ PHÂN LỚP QUYẾT ĐỊNH (2.0đ)
# ==============================================================
tree = id3(df, attrs)
print("Completed!\n" + "-" * 35)
H_D = entropy(df)
counts = df[target].value_counts().to_dict()
counts_str = ", ".join(f"{cnt} {lbl}" for lbl, cnt in counts.items())
print(f"Entropy tổng H(D) = {H_D:.4f} ({counts_str})")
print("-" * 35)
for a in attrs:
    print(f"Gain(D, {a:<9}) = {gain(df, a):.4f}")
best_root = max(attrs, key=lambda a: gain(df, a))
print(f"=> Nút gốc (Gain max): {best_root} = {gain(df, best_root):.4f}")
print("-" * 35)
show(tree)
print("-" * 35)
print(f"=> Quyết định ID3: {predict(tree, D)}")
