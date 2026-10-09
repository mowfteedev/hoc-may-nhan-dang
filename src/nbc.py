import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()
target = df.columns[-1]
attrs = list(df.columns[:-1])


# ---------- 1. Các hàm tính xác suất cơ bản ----------
def tien_nghiem(h):
    return (df[target] == h).mean()


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


# ---------- 2. Quyết định NBC: h_NBC = argmax P(h | D) ----------
def nbc(D):
    mau_so = xac_suat_mau(D)
    post = {h: (kha_nang(D, h) * tien_nghiem(h)) / mau_so for h in df[target].unique()}
    return max(post, key=post.get), post


# ---------- Chạy thử nghiệm ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"D = {D_mau}")

best, post = nbc(D_mau)
print("Completed!\n" + "-" * 35)
for j, (h, p) in enumerate(post.items(), 1):
    print(f"P(h{j}|D) = {p:.4f}")
print("-" * 35)
j_best = list(post).index(best) + 1
print(f"=> Quyết định NBC (h_NBC): h{j_best} = {best}")
