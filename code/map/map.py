import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()   # Đọc bảng và loại bỏ dòng thiếu dữ liệu
# df = df.drop(columns=["ID"])          # Bỏ cột ID/STT nếu có

target = df.columns[-1]                 # Cột cuối cùng là nhãn (không gian giả thuyết H)
attrs = list(df.columns[:-1])           # Các thuộc tính quan sát


# ---------- Xác suất tiên nghiệm P(h) ----------
def prior(val):
    return (df[target] == val).mean()


# ---------- Likelihood P(D_i | h) ----------
def likelihood(a, v, val):
    sub = df[df[target] == val]
    return (sub[a] == v).sum() / len(sub)


# ---------- Quyết định MAP: h_MAP = argmax P(h | D) ----------
def map_predict(D):
    classes = list(df[target].unique())
    score = {}
    h_map = {}

    for j, val in enumerate(classes, 1):
        h = f"h{j}"
        h_map[h] = val
        s = prior(val)
        print(f"\nGiả thuyết {h}: P({h}) = {s:.4f}")
        for i, a in enumerate(attrs, 1):
            p = likelihood(a, D[a], val)
            print(f"   P(D{i} | {h}) = {p:.4f}")
            s *= p                                   # nhân dồn: P(D|h) * P(h)
        score[h] = s
        print(f"   => P(D|{h}) * P({h}) = {s:.6f}")

    # Mẫu số P(D) = tổng các tử số (công thức xác suất toàn phần)
    P_D = sum(score.values())
    print(f"\n=> Xác suất mẫu P(D) = Σ [P(D|h) * P(h)] = {P_D:.6f}")

    # Thay công thức tính P(h | D)
    print("\n===== KẾT QUẢ P(h | D) =====")
    post = {}
    for h, s in score.items():
        post[h] = s / P_D
        print(f"P({h} | D) = {s:.6f} / {P_D:.6f} = {post[h]:.4f}")

    best = max(post, key=post.get)
    print(f"\nQuyết định MAP (h_MAP): {best} (lớp {h_map[best]})")
    return best, post


# ---------- Chạy thử nghiệm ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"D = {D_mau}")
best, post = map_predict(D_mau)
