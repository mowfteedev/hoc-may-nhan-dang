# ==============================================================
# TC1: KHAI BÁO THƯ VIỆN SỬ DỤNG CHO THUẬT TOÁN (0.5đ)
# ==============================================================
import io
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
# Công thức MLE: h_ML = argmax P(D|h) = argmax Π P(d_i|h)
# ==============================================================

# ==============================================================
# TC5: TÍNH THAM SỐ THỨ NHẤT - KHẢ NĂNG TỪNG ĐẶC TRƯNG P(D_i|h) (1.5đ)
# ==============================================================
def dac_trung(a, v, h):
    sub = df[df[target] == h]
    return (sub[a] == v).sum() / len(sub)


# ==============================================================
# TC6: TÍNH THAM SỐ THỨ HAI - TÍCH KHẢ NĂNG CẢ MẪU P(D|h) (1.5đ)
# ==============================================================
def kha_nang(D, h):
    prod = 1.0
    for a in attrs:
        prod *= dac_trung(a, D[a], h)
    return prod


def mle(D):
    L = {h: kha_nang(D, h) for h in df[target].unique()}
    return max(L, key=L.get), L


# ==============================================================
# TC7: NHẬP DỮ LIỆU TEST (1.0đ)
# ==============================================================
D = D_test
print(f"D = {D}")

# ==============================================================
# TC8 & TC9: PHÂN LỚP THỨ 1 & PHÂN LỚP THỨ 2 (2.0đ)
# ==============================================================
best, L = mle(D)
print("Completed!\n" + "-" * 35)
for j, (h, p) in enumerate(L.items(), 1):
    print(f"P(D|h{j}) = {p:.4f}")
print("-" * 35)
j_best = list(L).index(best) + 1
print(f"=> Quyết định MLE (h_ML): h{j_best} = {best}")
