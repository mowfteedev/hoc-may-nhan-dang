import pandas as pd

# ---------- Đọc dữ liệu ----------
df = pd.read_csv("data.csv").dropna()
# df = df.drop(columns=["ID"])          # bỏ cột ID/STT nếu có

target = df.columns[-1]                 # cột cuối là nhãn (loại A, ví dụ Banana/Orange/Other)
attrs = list(df.columns[:-1])           # các tính chất B1, B2, ..., Bn
ALPHA = 0                               # 0: tính tay như slide; 1: làm trơn Laplace (tránh xác suất 0)


# ---------- P(A): xác suất tiên nghiệm của lớp A ----------
# P(A) = số mẫu thuộc A / tổng số mẫu
def P_A(A):
    return (df[target] == A).mean()


# ---------- P(Bi | A): xác suất tính chất Bi = v khi biết lớp A ----------
# = (số mẫu A có Bi = v + ALPHA) / (số mẫu A + ALPHA * số giá trị của Bi)
def P_B_A(b, v, A):
    sub = df[df[target] == A]
    k = df[b].nunique()
    return ((sub[b] == v).sum() + ALPHA) / (len(sub) + ALPHA * k)


# ---------- P(Bi): xác suất riêng của Bi = v trên toàn bộ dữ liệu ----------
def P_B(b, v):
    return (df[b] == v).mean()


# ---------- Naive Bayes theo công thức slide ----------
# P(A|B) = [P(B1|A) x ... x P(Bn|A)] x P(A) / [P(B1) x ... x P(Bn)]
def naive_bayes(x, verbose=False):
    # Mẫu số: tích các P(Bi)
    mau_so = 1
    for b in attrs:
        p = P_B(b, x[b])
        if verbose:
            print(f"P({b}={x[b]}) = {p:.4f}")
        mau_so *= p
    if verbose:
        print(f"=> Mẫu số = {mau_so:.4f}\n")

    # Tử số của từng lớp rồi chia cho mẫu số
    tu_so, ket_qua = {}, {}
    for A in df[target].unique():
        t = P_A(A)
        if verbose:
            print(f"A = {A}: P({A}) = {t:.4f}")
        for b in attrs:
            p = P_B_A(b, x[b], A)
            if verbose:
                print(f"   P({b}={x[b]} | {A}) = {p:.4f}")
            t *= p
        tu_so[A] = t
        ket_qua[A] = t / mau_so if mau_so else 0
        if verbose:
            print(f"   => Tử số = {t:.4f}  =>  P({A} | B) = {t:.4f} / {mau_so:.4f} = {ket_qua[A]:.4f}\n")

    best = max(ket_qua, key=ket_qua.get)
    return best, ket_qua, tu_so


# ---------- Chạy ----------
# Quả cần phân loại: sửa theo tên cột và giá trị trong file CSV của bạn
mau = {"Long": "Yes", "Sweet": "Yes", "Yellow": "Yes"}
best, ket_qua, tu_so = naive_bayes(mau, verbose=True)

# Đối chiếu: mẫu số chuẩn hóa = tổng các tử số (tổng xác suất đúng bằng 1)
tong = sum(tu_so.values())
print("===== KẾT QUẢ =====")
print(f"{'A':<10}{'Theo slide':>12}{'Chuẩn hóa':>12}")
for A in ket_qua:
    print(f"{A:<10}{ket_qua[A]:>12.4f}{(tu_so[A] / tong if tong else 0):>12.4f}")
print("Quyết định:", best)

# Độ chính xác trên tập huấn luyện
pred = df.apply(lambda r: naive_bayes(r)[0], axis=1)
print(f"Độ chính xác (train): {(pred == df[target]).mean():.2%}")
