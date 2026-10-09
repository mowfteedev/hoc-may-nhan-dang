import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()   # Đọc bảng và loại bỏ dòng thiếu dữ liệu
# df = df.drop(columns=["ID"])          # Bỏ cột ID/STT nếu có

target = df.columns[-1]                 # Cột cuối cùng là nhãn (không gian giả thuyết H)
attrs = list(df.columns[:-1])           # Các thuộc tính quan sát


# ---------- Xác suất tiên nghiệm P(h) ----------
def prior(h):
    return (df[target] == h).mean()


# ---------- Likelihood P(d_i | h) ----------
# = (số mẫu của giả thuyết h có thuộc tính a = v) / (số mẫu của giả thuyết h)
def likelihood(a, v, h):
    sub = df[df[target] == h]
    return (sub[a] == v).sum() / len(sub)


# ---------- Quyết định MAP: h_MAP = argmax P(D|h) * P(h) ----------
def map_predict(D):
    score = {}
    for h in df[target].unique():
        s = prior(h)
        print(f"\nGiả thuyết {h}: P({h}) = {s:.4f}")
        for i, a in enumerate(attrs, 1):
            p = likelihood(a, D[a], h)
            print(f"   P(D{i} | {h}) = {p:.4f}")
            s *= p                                   # nhân dồn: P(D|h) * P(h)
        score[h] = s
        print(f"   => P({h}) * Π P(D_i|{h}) = {s:.6f}")

    total = sum(score.values())
    post = {h: s / total for h, s in score.items()}  # chuẩn hóa thành P(h | D)
    best = max(post, key=post.get)                   # argmax
    return best, post


# ---------- Chạy ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"D = {D_mau}")
best, post = map_predict(D_mau)

print("\n===== KẾT QUẢ =====")
for h, p in post.items():
    print(f"P({h} | D) = {p:.4f}")
print("Quyết định MAP (h_MAP):", best)
