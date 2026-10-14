# ==============================================================
# TC1: KHAI BÁO THƯ VIỆN SỬ DỤNG CHO THUẬT TOÁN (0.5đ)
# ==============================================================
import io
import pandas as pd

# ==============================================================
# TC2: TẠO DỮ LIỆU TRAIN RIÊNG CHO BÀI TOÁN (1.5đ)
# ==============================================================
train_data = """Outlook,Temp,Humidity,Wind,Play
Sunny,Hot,High,Weak,No
Sunny,Hot,High,Strong,No
Overcast,Hot,High,Weak,Yes
Rain,Mild,High,Weak,Yes
Rain,Cool,Normal,Weak,Yes
Rain,Cool,Normal,Strong,No
Overcast,Cool,Normal,Strong,Yes
Sunny,Mild,High,Weak,No
Sunny,Cool,Normal,Weak,Yes
Rain,Mild,Normal,Weak,Yes
Sunny,Mild,Normal,Strong,Yes
Overcast,Mild,High,Strong,Yes
Overcast,Hot,Normal,Weak,Yes
Rain,Mild,High,Strong,No"""

df = pd.read_csv(io.StringIO(train_data))
target = df.columns[-1]
attrs = list(df.columns[:-1])

# ==============================================================
# TC3: TẠO DỮ LIỆU TEST (1.0đ)
# ==============================================================
D_test = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}


# ==============================================================
# TC4: VIẾT ĐÚNG CÔNG THỨC SỬ DỤNG CHO THUẬT TOÁN (1.0đ)
# Công thức NBC: P(h|D) = [P(h) * Π P(d_i|h)] / Π P(d_i)
# ==============================================================

# ==============================================================
# TC5: TÍNH THAM SỐ THỨ NHẤT - TIÊN NGHIỆM P(h) (1.5đ)
# ==============================================================
def tien_nghiem(h):
    return (df[target] == h).mean()


# ==============================================================
# TC6: TÍNH THAM SỐ THỨ HAI - KHẢ NĂNG P(D|h) & XÁC SUẤT BIÊN P(D) (1.5đ)
# ==============================================================
def dac_trung(a, v, h):
    sub = df[df[target] == h]
    return (sub[a] == v).sum() / len(sub)


def kha_nang(D, h):
    prod = 1.0
    for a in attrs:
        prod *= dac_trung(a, D[a], h)
    return prod


def bien(a, v):
    return (df[a] == v).mean()


def xac_suat_mau(D):
    prod = 1.0
    for a in attrs:
        prod *= bien(a, D[a])
    return prod


def nbc(D):
    mau_so = xac_suat_mau(D)
    post = {h: (kha_nang(D, h) * tien_nghiem(h)) / mau_so for h in df[target].unique()}
    return max(post, key=post.get), post


# ==============================================================
# TC7: NHẬP DỮ LIỆU TEST (1.0đ)
# ==============================================================
D = D_test
print(f"D = {D}")

# ==============================================================
# TC8 & TC9: PHÂN LỚP THỨ 1 & PHÂN LỚP THỨ 2 (2.0đ)
# ==============================================================
best, post = nbc(D)
print("Completed!\n" + "-" * 35)
for j, (h, p) in enumerate(post.items(), 1):
    print(f"P(h{j}|D) = {p:.4f}")
print("-" * 35)
j_best = list(post).index(best) + 1
print(f"=> Quyết định NBC (h_NBC): h{j_best} = {best}")
