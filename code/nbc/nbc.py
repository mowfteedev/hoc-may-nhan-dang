import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()   # Đọc bảng và loại bỏ dòng thiếu dữ liệu
# df = df.drop(columns=["ID"])          # Bỏ cột ID/STT nếu có

target = df.columns[-1]                 # Cột cuối cùng là nhãn cần dự đoán (lớp c)
attrs = list(df.columns[:-1])           # Các thuộc tính đặc trưng (features)
values = {a: df[a].unique() for a in attrs}   # Tập giá trị có thể của từng thuộc tính
ALPHA = 0                               # 0: tính tay như slide; 1: làm trơn Laplace (tránh xác suất 0)


# ---------- Xác suất tiên nghiệm: P(c) ----------
# P(c) = số mẫu thuộc lớp c / tổng số mẫu
def prior(c):
    return (df[target] == c).mean()


# ---------- Khả năng (Likelihood): P(a = v | c) ----------
# = (số mẫu lớp c có thuộc tính a = v + ALPHA) / (số mẫu lớp c + ALPHA * số giá trị của a)
def likelihood(a, v, c):
    sub = df[df[target] == c]
    k = len(values[a])
    return ((sub[a] == v).sum() + ALPHA) / (len(sub) + ALPHA * k)


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
    numerator, posterior = {}, {}
    for c in df[target].unique():
        num = prior(c)
        if verbose:
            print(f"Lớp {c}: P({c}) = {num:.4f}")
        for a in attrs:
            p = likelihood(a, x[a], c)
            if verbose:
                print(f"   P({a}={x[a]} | {c}) = {p:.4f}")
            num *= p
        numerator[c] = num
        posterior[c] = num / denominator if denominator else 0.0
        if verbose:
            print(f"   => Tử số = {num:.4f}  =>  P({c} | x) = {num:.4f} / {denominator:.4f} = {posterior[c]:.4f}\n")

    best = max(posterior, key=posterior.get)          # chọn lớp có hậu nghiệm lớn nhất
    return best, posterior, numerator


# ---------- Chạy thử nghiệm ----------
# Mẫu cần phân loại (quan sát mới)
x_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
best, posterior, numerator = nbc_predict(x_mau, verbose=True)

# Đối chiếu: Mẫu số chuẩn hóa = tổng các tử số (tổng xác suất đúng bằng 1)
total = sum(numerator.values())
print("===== KẾT QUẢ PHÂN LỚP NBC =====")
print(f"{'Lớp':<10}{'Theo slide':>12}{'Chuẩn hóa':>12}")
for c in posterior:
    normalized = (numerator[c] / total) if total else 0.0
    print(f"{str(c):<10}{posterior[c]:>12.4f}{normalized:>12.4f}")
print("Quyết định NBC (c_NBC):", best)

# Đánh giá độ chính xác trên tập huấn luyện
pred = df.apply(lambda r: nbc_predict(r)[0], axis=1)
print(f"\nĐộ chính xác (train): {(pred == df[target]).mean():.2%}")
