# ==============================================================
# TC1: KHAI BÁO THƯ VIỆN SỬ DỤNG CHO THUẬT TOÁN (0.5đ)
# ==============================================================
import math
from collections import defaultdict

# ==============================================================
# TC2: TẠO DỮ LIỆU TRAIN RIÊNG CHO BÀI TOÁN (1.5đ)
# ==============================================================
dulieu = [
    {"Ngay": 1, "Tiet_troi": "Nang", "Nhiet_do": "Nong", "Do_am": "Cao", "Gio": "Yeu", "Choicau": "Khong"},
    {"Ngay": 2, "Tiet_troi": "Nang", "Nhiet_do": "Nong", "Do_am": "Cao", "Gio": "Manh", "Choicau": "Khong"},
    {"Ngay": 3, "Tiet_troi": "Nhieu_may", "Nhiet_do": "Nong", "Do_am": "Cao", "Gio": "Yeu", "Choicau": "Co"},
    {"Ngay": 4, "Tiet_troi": "Mua", "Nhiet_do": "Mat", "Do_am": "Cao", "Gio": "Yeu", "Choicau": "Co"},
    {"Ngay": 5, "Tiet_troi": "Mua", "Nhiet_do": "Lanh", "Do_am": "Thuong", "Gio": "Yeu", "Choicau": "Co"},
    {"Ngay": 6, "Tiet_troi": "Mua", "Nhiet_do": "Lanh", "Do_am": "Thuong", "Gio": "Manh", "Choicau": "Khong"},
    {"Ngay": 7, "Tiet_troi": "Nhieu_may", "Nhiet_do": "Lanh", "Do_am": "Thuong", "Gio": "Manh", "Choicau": "Co"},
    {"Ngay": 8, "Tiet_troi": "Nang", "Nhiet_do": "Mat", "Do_am": "Cao", "Gio": "Yeu", "Choicau": "Khong"},
    {"Ngay": 9, "Tiet_troi": "Nang", "Nhiet_do": "Lanh", "Do_am": "Thuong", "Gio": "Yeu", "Choicau": "Co"},
    {"Ngay": 10, "Tiet_troi": "Mua", "Nhiet_do": "Mat", "Do_am": "Thuong", "Gio": "Yeu", "Choicau": "Co"},
    {"Ngay": 11, "Tiet_troi": "Nang", "Nhiet_do": "Mat", "Do_am": "Thuong", "Gio": "Manh", "Choicau": "Co"},
    {"Ngay": 12, "Tiet_troi": "Nhieu_may", "Nhiet_do": "Mat", "Do_am": "Cao", "Gio": "Manh", "Choicau": "Co"},
    {"Ngay": 13, "Tiet_troi": "Nhieu_may", "Nhiet_do": "Nong", "Do_am": "Thuong", "Gio": "Yeu", "Choicau": "Co"},
    {"Ngay": 14, "Tiet_troi": "Mua", "Nhiet_do": "Mat", "Do_am": "Cao", "Gio": "Manh", "Choicau": "Khong"},
]

target = "Choicau"
attrs = ["Tiet_troi", "Nhiet_do", "Do_am", "Gio"]

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
    if not D:
        return 0.0
    dem = defaultdict(int)
    for m in D:
        dem[m[target]] += 1
    tong = len(D)
    return -sum((c / tong) * math.log2(c / tong) for c in dem.values() if c > 0)


# ==============================================================
# TC6: TÍNH THAM SỐ THỨ HAI CỦA THUẬT TOÁN - GAIN & DỰNG CÂY (1.5đ)
# ==============================================================
def gain(D, a):
    nhom = defaultdict(list)
    for m in D:
        nhom[m[a]].append(m)
    tong = len(D)
    return entropy(D) - sum((len(D_v) / tong) * entropy(D_v) for D_v in nhom.values())


def id3(D, current_attrs):
    nhan = [m[target] for m in D]
    nhan_da_so = max(set(nhan), key=nhan.count)

    # Điều kiện dừng: Thuần nhất hoặc hết thuộc tính chia
    if len(set(nhan)) == 1:
        return nhan[0]
    if not current_attrs:
        return nhan_da_so

    # Chọn thuộc tính có Gain lớn nhất
    best = max(current_attrs, key=lambda a: gain(D, a))
    rest = [a for a in current_attrs if a != best]

    tree = {}
    gia_tri_goc = list(dict.fromkeys(m[best] for m in dulieu))
    for v in gia_tri_goc:
        D_v = [m for m in D if m[best] == v]
        tree[v] = id3(D_v, rest) if D_v else nhan_da_so
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
    return predict(tree[a].get(D.get(a)), D)


# ==============================================================
# TC8: DỰNG CÂY VÀ TÍNH THAM SỐ (1.0đ)
# ==============================================================
tree = id3(dulieu, attrs)
print("Completed!\n" + "-" * 35)

H_D = entropy(dulieu)
dem_nhan = defaultdict(int)
for m in dulieu:
    dem_nhan[m[target]] += 1
counts_str = ", ".join(f"{cnt} {lbl}" for lbl, cnt in sorted(dem_nhan.items(), key=lambda x: -x[1]))
print(f"Entropy tổng H(D) = {H_D:.4f} ({counts_str})")
print("-" * 35)

for a in attrs:
    print(f"Gain(D, {a:<9}) = {gain(dulieu, a):.4f}")
best_root = max(attrs, key=lambda a: gain(dulieu, a))
print(f"=> Nút gốc (Gain max): {best_root} = {gain(dulieu, best_root):.4f}")
print("-" * 35)

show(tree)
print("-" * 35)

# ==============================================================
# TC7 & TC9: NHẬP DỮ LIỆU TEST VÀ DỰ ĐOÁN QUYẾT ĐỊNH (2.0đ)
# ==============================================================
D = D_test
print(f"D = {D}")
print(f"=> Quyết định ID3: {predict(tree, D)}")
