import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()   # Đọc bảng và loại bỏ dòng thiếu dữ liệu
# df = df.drop(columns=["ID"])          # Bỏ cột ID/STT nếu có

target = df.columns[-1]                 # Cột cuối cùng là nhãn cần dự đoán (lớp c)
attrs = list(df.columns[:-1])           # Các thuộc tính đặc trưng (features)


# ---------- Xác suất tiên nghiệm: P(c) ----------
# P(c) = số mẫu thuộc lớp c / tổng số mẫu
def prior(c):
    return (df[target] == c).mean()


# ---------- Khả năng (Likelihood): P(a = v | c) ----------
# = (số mẫu lớp c có thuộc tính a = v) / (số mẫu lớp c)
def likelihood(a, v, c):
    sub = df[df[target] == c]
    return (sub[a] == v).sum() / len(sub)


# ---------- Xác suất biên (Evidence): P(a = v) ----------
# Xác suất xuất hiện giá trị v của thuộc tính a trên toàn bộ dữ liệu
def evidence(a, v):
    return (df[a] == v).mean()


# ---------- Phân lớp Naive Bayes (NBC): P(c | x) ----------
# P(c|x) = [P(c) * Π P(x_i | c)] / Π P(x_i)
def nbc_predict(x, verbose=False):
    # Mẫu số: tích các xác suất biên của từng thuộc tính Π P(x_i)
    denominator = 1.0
    for a in attrs:
        p = evidence(a, x[a])
        if verbose:
            print(f"P({a}={x[a]}) = {p:.4f}")
        denominator *= p
    if verbose:
        print(f"=> Mẫu số Π P(x_i) = {denominator:.4f}\n")

    # Tử số của từng lớp: P(c) * Π P(x_i | c)
    posterior = {}
    for c in df[target].unique():
        num = prior(c)
        if verbose:
            print(f"Lớp {c}: P({c}) = {num:.4f}")
        for a in attrs:
            p = likelihood(a, x[a], c)
            if verbose:
                print(f"   P({a}={x[a]} | {c}) = {p:.4f}")
            num *= p
        post = num / denominator if denominator else 0.0
        posterior[c] = post
        if verbose:
            print(f"   => Tử số = {num:.4f}  =>  P({c} | x) = {num:.4f} / {denominator:.4f} = {post:.4f}\n")

    best = max(posterior, key=posterior.get)          # chọn lớp có hậu nghiệm lớn nhất
    return best, posterior


# ---------- Chạy thử nghiệm ----------
# Mẫu cần phân loại (quan sát mới)
x_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
best, posterior = nbc_predict(x_mau, verbose=True)

print("===== KẾT QUẢ PHÂN LỚP NBC =====")
for c, p in posterior.items():
    print(f"P({c} | x) = {p:.4f}")
print("Quyết định NBC (c_NBC):", best)

# Đánh giá độ chính xác trên tập huấn luyện
pred = df.apply(lambda r: nbc_predict(r)[0], axis=1)
print(f"\nĐộ chính xác (train): {(pred == df[target]).mean():.2%}")
