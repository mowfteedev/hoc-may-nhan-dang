import pandas as pd

# ---------- Dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()
target, attrs = df.columns[-1], list(df.columns[:-1])
ALPHA = 0                               # 0: tính tay như slide; 1: làm trơn Laplace


# ---------- 1. Tiên nghiệm P(h) ----------
def P_h(h):
    return (df[target] == h).mean()


# ---------- 2. Khả năng P(d_i | h) ----------
def P_di_h(a, v, h):
    sub = df[df[target] == h]
    return ((sub[a] == v).sum() + ALPHA) / (len(sub) + ALPHA * df[a].nunique())


# ---------- 3. Quyết định MAP ----------
def map_predict(D, verbose=False):
    score = {}
    for h in df[target].unique():
        s = P_h(h)
        if verbose: print(f"Giả thuyết {h}: P({h}) = {s:.4f}")
        for a in attrs:
            p = P_di_h(a, D[a], h)
            if verbose: print(f"   P({a}={D[a]} | {h}) = {p:.4f}")
            s *= p
        score[h] = s
        if verbose: print(f"   => P({h}) * Π P(d_i|{h}) = {s:.6f}\n")

    total = sum(score.values())
    post = {h: s / total for h, s in score.items()} if total else score
    return max(post, key=post.get), post, score


# ---------- Chạy & Đánh giá ----------
print("===== 1. CHI TIẾT TÍNH TOÁN CHO MẪU D =====")
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"Mẫu quan sát D: {D_mau}\n")
best, post, score = map_predict(D_mau, verbose=True)

print("===== 2. KẾT QUẢ ĐÁNH GIÁ & DỰ ĐOÁN =====")
for h, p in post.items():
    print(f"P({h} | D) = {p:.4f} (Điểm số chưa chuẩn hóa: {score[h]:.6f})")

pred = df.apply(lambda r: map_predict(r)[0], axis=1)
acc = (pred == df[target]).mean()
correct = (pred == df[target]).sum()
print(f"\nĐộ chính xác trên tập huấn luyện: {acc:.2%} ({correct}/{len(df)} mẫu đúng)")
print(f"=> Quyết định MAP cho mẫu D (h_MAP): {best}")
