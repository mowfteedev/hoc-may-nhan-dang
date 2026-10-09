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


def xac_suat_mau(D):
    return sum(kha_nang(D, h) * tien_nghiem(h) for h in df[target].unique())


# ---------- 2. Quyết định MAP: h_MAP = argmax P(h | D) ----------
def hau_nghiem(D):
    classes = list(df[target].unique())
    mau_so = xac_suat_mau(D)
    post = {f"h{j}": (kha_nang(D, h) * tien_nghiem(h)) / mau_so for j, h in enumerate(classes, 1)}
    best = max(post, key=post.get)
    return best, post


# ---------- Chạy thử nghiệm ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"D = {D_mau}")

best, post = hau_nghiem(D_mau)
print("Completed!")
for h, p in post.items():
    print(f"P({h} | D) = {p:.4f}")
print(f"Quyết định MAP (h_MAP): {best}")
