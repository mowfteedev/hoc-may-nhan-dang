import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()   # Đọc bảng và loại bỏ dòng thiếu dữ liệu
# df = df.drop(columns=["ID"])          # Bỏ cột ID/STT nếu có

target = df.columns[-1]                 # Cột cuối cùng là nhãn (không gian giả thuyết H)
attrs = list(df.columns[:-1])           # Các thuộc tính quan sát
ALPHA = 0                               # 0: tính tay như sách; 1: làm trơn Laplace (tránh xác suất 0)


# ---------- Xác suất tiên nghiệm P(h) ----------
def prior(h):
    return (df[target] == h).mean()


# ---------- Likelihood P(d_i | h) ----------
# = (số mẫu của giả thuyết h có thuộc tính a = v) / (số mẫu của giả thuyết h)
def likelihood(a, v, h):
    sub = df[df[target] == h]
    k = df[a].nunique()                 # số giá trị của thuộc tính (dùng cho Laplace)
    return ((sub[a] == v).sum() + ALPHA) / (len(sub) + ALPHA * k)


# ---------- Quyết định MAP: h_MAP = argmax P(D|h) * P(h) ----------
def map_predict(D):
    score = {}
    for h in df[target].unique():
        s = prior(h)
        print(f"\nGiả thuyết {h}: P({h}) = {s:.4f}")
        for a in attrs:
            p = likelihood(a, D[a], h)
            print(f"   P({a}={D[a]} | {h}) = {p:.4f}")
            s *= p                                   # nhân dồn: P(D|h) * P(h)
        score[h] = s
        print(f"   => P({h}) * Π P(d_i|{h}) = {s:.6f}")

    total = sum(score.values())
    post = {h: s / total for h, s in score.items()}  # chuẩn hóa thành P(h | D)
    best = max(post, key=post.get)                   # argmax
    return best, post


# ---------- Chạy ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
best, post = map_predict(D_mau)

print("\n===== KẾT QUẢ =====")
for h, p in post.items():
    print(f"P({h} | D) = {p:.4f}")
print("Quyết định MAP (h_MAP):", best)
