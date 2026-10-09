import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()
target = df.columns[-1]
attrs = list(df.columns[:-1])


# ---------- 1. Các hàm tính xác suất cơ bản ----------
def dac_trung(a, v, h):
    sub = df[df[target] == h]
    return (sub[a] == v).sum() / len(sub)


def kha_nang(D, h):
    prod = 1.0
    for a in attrs:
        prod *= dac_trung(a, D[a], h)
    return prod


# ---------- 2. Quyết định MLE: h_ML = argmax P(D | h) ----------
def mle(D):
    L, labels = {}, {}
    for j, h in enumerate(df[target].unique(), 1):
        L[f"h{j}"] = kha_nang(D, h)
        labels[f"h{j}"] = h
    best = max(L, key=L.get)
    return f"{best} = {labels[best]}", L


# ---------- Chạy thử nghiệm ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"D = {D_mau}")

best, L = mle(D_mau)
print("Completed!\n" + "-" * 35)
for h, p in L.items():
    print(f"P(D|{h}) = {p:.4f}")
print("-" * 35)
print(f"=> Quyết định MLE (h_ML): {best}")
