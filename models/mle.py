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
    L = {h: kha_nang(D, h) for h in df[target].unique()}
    return max(L, key=L.get), L


# ---------- Chạy thử nghiệm ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"D = {D_mau}")

best, L = mle(D_mau)
print("Completed!\n" + "-" * 35)
for j, (h, p) in enumerate(L.items(), 1):
    print(f"P(D|h{j}) = {p:.4f}")
print("-" * 35)
j_best = list(L).index(best) + 1
print(f"=> Quyết định MLE (h_ML): h{j_best} = {best}")
