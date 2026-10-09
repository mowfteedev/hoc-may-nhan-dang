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
    # Tử số: P(D|h) * P(h)
    score = {h: P_D_h(D, h) * P_h(h) for h in classes}
    # Mẫu số: P(D) = Σ [P(D|h) * P(h)]
    P_D = sum(score.values())
    # Hậu nghiệm: P(h | D)
    post = {f"h{j}": score[h] / P_D for j, h in enumerate(classes, 1)}
    best = max(post, key=post.get)
    return best, post


# ---------- Chạy thử nghiệm ----------
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"D = {D_mau}")

best, post = map_predict(D_mau)
print("Completed!")
for h, p in post.items():
    print(f"P({h} | D) = {p:.4f}")
print(f"Quyết định MAP (h_MAP): {best}")
