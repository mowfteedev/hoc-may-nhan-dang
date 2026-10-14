import os
import math
import pandas as pd

# 1. Đọc dữ liệu
DATA_FILE = "data/data.csv" if os.path.exists("data/data.csv") else os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
df = pd.read_csv(DATA_FILE)
target = df.columns[-1]
attrs = list(df.columns[:-1])

D_test = {"Tiet_troi": "Nang", "Nhiet_do": "Lanh", "Do_am": "Cao", "Gio": "Manh"}


# 2. Các hàm cốt lõi ID3
def entropy(D):
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p if x > 0)


def gain(D, a):
    w = D[a].value_counts(normalize=True)
    e = sum(w[v] * entropy(sub) for v, sub in D.groupby(a))
    return entropy(D) - e


def id3(D, attrs):
    """Tính Gain cho từng thuộc tính và sắp xếp giảm dần"""
    gains = [(gain(D, a), a) for a in attrs]
    gains.sort(reverse=True)
    return [a for g, a in gains]


def build(D, attrs):
    """Dựng cây quyết định ID3 đệ quy chuẩn mực"""
    if D[target].nunique() == 1 or not attrs:
        return D[target].mode()[0]

    order = id3(D, attrs)
    best = order[0]
    rest = order[1:]

    tree = {}
    for v, sub in D.groupby(best):
        tree[v] = build(sub, rest)
    return {best: tree}


# 3. Hàm hiển thị cây & dự đoán
def show(tree, prefix="", is_root=True):
    """Vẽ cây quyết định dạng cây thư mục Unicode"""
    if not isinstance(tree, dict):
        return
    attr = next(iter(tree))
    if is_root:
        print(attr)
    items = list(tree[attr].items())
    for i, (val, sub) in enumerate(items):
        is_last = (i == len(items) - 1)
        branch = "└── " if is_last else "├── "
        next_prefix = prefix + ("    " if is_last else "│   ")
        if isinstance(sub, dict):
            child_attr = next(iter(sub))
            child_items = list(sub[child_attr].items())
            if all(not isinstance(v, dict) for _, v in child_items):
                print(f"{prefix}{branch}{val}")
                for j, (cv, csub) in enumerate(child_items):
                    c_is_last = (j == len(child_items) - 1)
                    c_branch = "└── " if c_is_last else "├── "
                    print(f"{next_prefix}{c_branch}{child_attr} = {cv} -> {csub}")
            else:
                print(f"{prefix}{branch}{val} -> {child_attr}")
                show(sub, next_prefix, is_root=False)
        else:
            print(f"{prefix}{branch}{val} -> {sub}")


def predict(tree, sample, default_label=None):
    """Dự đoán nhãn cho mẫu kiểm thử"""
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    val = sample.get(attr)
    subtree = tree[attr].get(val, default_label or df[target].mode()[0])
    return predict(subtree, sample, default_label)


# 4. Thực thi và in kết quả chuẩn barem chấm điểm
tree = build(df, attrs)
print("Completed!\n" + "-" * 35)

H_D = entropy(df)
counts = df[target].value_counts().to_dict()
counts_str = ", ".join(f"{cnt} {lbl}" for lbl, cnt in counts.items())
print(f"Entropy tổng H(D) = {H_D:.4f} ({counts_str})")
print("-" * 35)

# Danh sách thuộc tính xếp theo Gain từ cao xuống thấp
ranked_attrs = [(gain(df, a), a) for a in attrs]
ranked_attrs.sort(reverse=True)
for g, a in ranked_attrs:
    print(f"Gain(D, {a}) = {g:.4f}")

best_root = ranked_attrs[0][1]
print(f"=> Nút gốc (Gain max): {best_root} = {ranked_attrs[0][0]:.4f}")
print("-" * 35)

show(tree)
print("-" * 35)

D = D_test
print(f"D = {D}")
print(f"=> Quyết định ID3: {predict(tree, D)}")

