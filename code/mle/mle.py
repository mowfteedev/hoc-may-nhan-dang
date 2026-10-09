import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()   # Đọc bảng và loại bỏ dòng thiếu dữ liệu
# df = df.drop(columns=["ID"])          # Bỏ cột ID/STT nếu có

target = df.columns[-1]                 # Cột cuối cùng là nhãn (không gian giả thuyết H)
attrs = list(df.columns[:-1])           # Các thuộc tính quan sát


# ---------- Ước lượng Likelihood: P(d_i = v | h) ----------
# Likelihood của biến rời rạc = số lần xuất hiện / tổng số mẫu của giả thuyết h
def likelihood(a, v, h):
    sub = df[df[target] == h]
    return (sub[a] == v).sum() / len(sub)


# ---------- Quyết định MLE: h_ML = argmax P(D|h) ----------
def mle_predict(D):
    L = {}
    for h in df[target].unique():
        s = 1.0
        print(f"\nGiả thuyết {h}:")
        for a in attrs:
            p = likelihood(a, D[a], h)
            print(f"   P({a}={D[a]} | {h}) = {p:.4f}")
            s *= p                                   # nhân dồn likelihood: P(D|h)
        L[h] = s
        print(f"   => L({h}) = P(D|{h}) = Π P(d_i|{h}) = {s:.6f}")
    best = max(L, key=L.get)                         # argmax likelihood
    return best, L


# ---------- Chạy ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
best, L = mle_predict(D_mau)

print("\n===== KẾT QUẢ =====")
for h, v in L.items():
    print(f"L({h}) = {v:.6f}")
print("Quyết định MLE (h_ML):", best)
