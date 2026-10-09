import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()
target = df.columns[-1]
attrs = list(df.columns[:-1])


# ---------- 1. Các hàm tính xác suất cơ bản ----------
def P_h(h):
    return (df[target] == h).mean()


def P_Di_h(a, v, h):
    sub = df[df[target] == h]
    return (sub[a] == v).sum() / len(sub)


def P_D_h(D, h):
    prod = 1.0
    for a in attrs:
        prod *= P_Di_h(a, D[a], h)
    return prod


# ---------- 2. Quyết định MAP: h_MAP = argmax P(h | D) ----------
def map_predict(D):
    classes = list(df[target].unique())
    score = {}

    for j, h in enumerate(classes, 1):
        print(f"\nGiả thuyết h{j}: P(h{j}) = {P_h(h):.4f}")
        for i, a in enumerate(attrs, 1):
            print(f"   P(D{i} | h{j}) = {P_Di_h(a, D[a], h):.4f}")
        score[h] = P_D_h(D, h) * P_h(h)
        print(f"   => P(D|h{j}) * P(h{j}) = {score[h]:.6f}")

    # Mẫu số: P(D) = Σ [P(D|h) * P(h)]
    P_D = sum(score.values())
    print(f"\n=> Xác suất mẫu P(D) = Σ [P(D|h) * P(h)] = {P_D:.6f}")

    # Hậu nghiệm: P(h | D) = tử số / P(D)
    print("\n===== KẾT QUẢ P(h | D) =====")
    post = {}
    for j, h in enumerate(classes, 1):
        post[h] = score[h] / P_D
        print(f"P(h{j} | D) = {score[h]:.6f} / {P_D:.6f} = {post[h]:.4f}")

    best = max(post, key=post.get)
    best_idx = classes.index(best) + 1
    print(f"\nQuyết định MAP (h_MAP): h{best_idx}")
    return best, post


# ---------- Chạy thử nghiệm ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"D = {D_mau}")
best, post = map_predict(D_mau)
