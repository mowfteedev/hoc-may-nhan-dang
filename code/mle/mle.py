import pandas as pd

# ---------- Dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()
target, attrs = df.columns[-1], list(df.columns[:-1])


# ---------- 1. Khả năng Likelihood P(d_i | h) ----------
def P_di_h(a, v, h):
    sub = df[df[target] == h]
    return (sub[a] == v).sum() / len(sub)


# ---------- 2. Quyết định MLE: h_ML = argmax Π P(d_i | h) ----------
def mle_predict(D, verbose=False):
    L = {}
    for h in df[target].unique():
        s = 1.0
        if verbose: print(f"Giả thuyết {h}:")
        for a in attrs:
            p = P_di_h(a, D[a], h)
            if verbose: print(f"   P({a}={D[a]} | {h}) = {p:.4f}")
            s *= p
        L[h] = s
        if verbose: print(f"   => L({h}) = P(D|{h}) = Π P(d_i|{h}) = {s:.6f}\n")
    return max(L, key=L.get), L


# ---------- Chạy & Đánh giá ----------
print("===== 1. CHI TIẾT TÍNH TOÁN LIKELIHOOD CHO MẪU D =====")
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"Mẫu quan sát D: {D_mau}\n")
best, L = mle_predict(D_mau, verbose=True)

print("===== 2. KẾT QUẢ ĐÁNH GIÁ & QUYẾT ĐỊNH =====")
for h, v in L.items():
    print(f"L({h}) = P(D | {h}) = {v:.6f}")

pred = df.apply(lambda r: mle_predict(r)[0], axis=1)
acc = (pred == df[target]).mean()
correct = (pred == df[target]).sum()
print(f"\nĐộ chính xác trên tập huấn luyện: {acc:.2%} ({correct}/{len(df)} mẫu đúng)")
print(f"=> Quyết định MLE cho mẫu D (h_ML): {best}")
