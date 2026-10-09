import pandas as pd

# ---------- Dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()
target, attrs = df.columns[-1], list(df.columns[:-1])
values = {a: df[a].unique() for a in attrs}
ALPHA = 0                               # 0: tính tay như slide; 1: làm trơn Laplace


# ---------- 1. Tiên nghiệm P(h) ----------
def P_h(h):
    return (df[target] == h).mean()


# ---------- 2. Khả năng P(d_i | h) ----------
def P_di_h(a, v, h):
    sub = df[df[target] == h]
    k = len(values[a])
    return ((sub[a] == v).sum() + ALPHA) / (len(sub) + ALPHA * k)


# ---------- 3. Quyết định MAP: h_MAP = argmax P(h) * Π P(d_i | h) ----------
def map_predict(D, verbose=False):
    score = {}
    for h in df[target].unique():
        s = P_h(h)
        if verbose:
            print(f"Giả thuyết {h}: P({h}) = {s:.4f}")
        for a in attrs:
            p = P_di_h(a, D[a], h)
            if verbose:
                print(f"   P({a}={D[a]} | {h}) = {p:.4f}")
            s *= p
        score[h] = s
        if verbose:
            print(f"   => P({h}) * Π P(d_i|{h}) = {s:.6f}\n")

    total = sum(score.values())
    post = {h: s / total for h, s in score.items()} if total else {h: 0 for h in score}
    best = max(post, key=post.get)
    return best, post, score


# ==============================================================
# CHẠY VÀ ĐÁNH GIÁ (ĐÚNG 4 MỤC RÕ RÀNG)
# ==============================================================

# 1. Tiên nghiệm P(h)
print("===== 1. XÁC SUẤT TIÊN NGHIỆM P(h) =====")
for h in df[target].unique():
    print(f"P({h}) = {P_h(h):.4f}")

# 2. Chi tiết tính toán cho mẫu thử nghiệm D_mau
print("\n===== 2. CHI TIẾT TÍNH TOÁN CHO MẪU D =====")
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"Mẫu quan sát D: {D_mau}\n")
best, post, score = map_predict(D_mau, verbose=True)

# 3. Bảng xác suất hậu nghiệm
print("===== 3. BẢNG XÁC SUẤT HẬU NGHIỆM P(h | D) =====")
for h, p in post.items():
    print(f"P({h} | D) = {p:.4f} (Điểm số chưa chuẩn hóa: {score[h]:.6f})")

# 4. Độ chính xác và Quyết định
print("\n===== 4. ĐỘ CHÍNH XÁC VÀ QUYẾT ĐỊNH =====")
pred = df.apply(lambda r: map_predict(r)[0], axis=1)
accuracy = (pred == df[target]).mean()
correct = (pred == df[target]).sum()
print(f"Độ chính xác trên tập huấn luyện: {accuracy:.2%} ({correct}/{len(df)} mẫu đúng)")
print(f"Quyết định MAP cho mẫu D (h_MAP): {best}")
