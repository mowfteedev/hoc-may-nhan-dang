"""
DANH SÁCH CÁC HÀM CƠ BẢN DÙNG CHUNG CHO CÁC THUẬT TOÁN HỌC MÁY
================================================================
Mục đích:
- Chuẩn hóa các bước tính toán giống nhau giữa các bài toán (ID3, MLE, MAP, NBC).
- Phục vụ ôn tập, tra cứu nhanh trong phòng thi thực hành.
- Giữ nguyên lý DRY: các bước cơ bản dùng chung 1 hàm, từng thuật toán chỉ mở rộng
  logic đầu ra (Output / Decision rule) đặc thù.
================================================================
"""

import pandas as pd


# ==============================================================
# 1. BƯỚC DÙNG CHUNG: ĐỌC & TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================
def load_data(filepath="data/datatemplate.csv"):
    """
    Đọc dữ liệu CSV và tự động bóc tách cấu trúc:
    - target: Cột cuối cùng (nhãn phân lớp / giả thuyết)
    - attrs : Danh sách các cột đặc trưng (features)
    - values: Dict chứa tập tất cả các giá trị duy nhất của từng thuộc tính
    """
    df = pd.read_csv(filepath).dropna()
    # df = df.drop(columns=["ID"])  # Mở comment nếu dữ liệu có cột ID/STT

    target = df.columns[-1]
    attrs = list(df.columns[:-1])
    values = {a: df[a].unique() for a in attrs}
    return df, target, attrs, values


# ==============================================================
# 2. BƯỚC DÙNG CHUNG: XÁC SUẤT TIÊN NGHIỆM P(c)
# ==============================================================
def prior(df, target, c):
    """
    Xác suất tiên nghiệm của lớp c: P(c)
    - Dùng trong: MAP, NBC
    - MLE xem như phân phối đều (hoặc = 1) nên không cần dùng prior.
    """
    return (df[target] == c).mean()


# ==============================================================
# 3. BƯỚC DÙNG CHUNG: KHẢ NĂNG (LIKELIHOOD) P(a = v | c)
# ==============================================================
def likelihood(df, target, values, a, v, c, alpha=0):
    """
    Tính P(thuộc tính a = giá trị v | lớp c)
    Công thức tổng quát với làm trơn Laplace:
        P = (số mẫu lớp c có a = v + alpha) / (tổng số mẫu lớp c + alpha * |V_a|)

    Quy ước alpha khi thi:
    - alpha = 0: Tính tay theo slide lý thuyết hoặc chuẩn của MLE.
    - alpha = 1: Làm trơn Laplace (tránh xác suất 0 khi gặp giá trị chưa từng thấy).
    """
    sub = df[df[target] == c]
    k = len(values[a])
    count = (sub[a] == v).sum()
    return (count + alpha) / (len(sub) + alpha * k)


# ==============================================================
# 4. BƯỚC DÙNG CHO NBC: XÁC SUẤT BIÊN P(a = v) (EVIDENCE)
# ==============================================================
def evidence(df, a, v):
    """
    Xác suất biên của thuộc tính a = v trên toàn bộ tập dữ liệu.
    Dùng cho mẫu số của công thức Naive Bayes theo slide: Π P(x_i)
    """
    return (df[a] == v).mean()


# ==============================================================
# 5. BƯỚC DÙNG CHUNG: TÍCH KHẢ NĂNG (JOINT LIKELIHOOD PRODUCT)
# ==============================================================
def joint_likelihood(df, target, values, attrs, x, c, alpha=0, verbose=False):
    """
    Tính tích các likelihood của tất cả thuộc tính trong mẫu x với lớp c:
        Π P(a_i = x[a_i] | c)
    Đây là bước lõi giống hệt nhau ở cả 3 thuật toán: MLE, MAP, NBC.
    """
    prod = 1.0
    for a in attrs:
        p = likelihood(df, target, values, a, x[a], c, alpha)
        if verbose:
            print(f"   P({a}={x[a]} | {c}) = {p:.4f}")
        prod *= p
    return prod


# ==============================================================
# 6. BƯỚC DÙNG CHO NBC: MẪU SỐ TÍCH CÁC XÁC SUẤT BIÊN
# ==============================================================
def joint_evidence(df, attrs, x, verbose=False):
    """
    Mẫu số riêng của Naive Bayes theo slide: Π P(x_i)
    """
    prod = 1.0
    for a in attrs:
        p = evidence(df, a, x[a])
        if verbose:
            print(f"P({a}={x[a]}) = {p:.4f}")
        prod *= p
    return prod


# ==============================================================
# 7. BƯỚC DÙNG CHUNG: ĐÁNH GIÁ ĐỘ CHÍNH XÁC (ACCURACY)
# ==============================================================
def evaluate_accuracy(predict_fn, df, target):
    """
    Đánh giá độ chính xác của hàm phân loại trên tập dữ liệu:
    - Dùng chung cho: ID3, MLE, MAP, NBC
    predict_fn: hàm nhận vào dòng dữ liệu (Series/dict) và trả về nhãn dự đoán (best)
    """
    pred = df.apply(lambda r: predict_fn(r)[0] if isinstance(predict_fn(r), (tuple, list)) else predict_fn(r), axis=1)
    acc = (pred == df[target]).mean()
    return acc


# ==============================================================
# 8. BẢNG TRA CỨU ĐẦU RA (OUTPUT) TỪNG THUẬT TOÁN KHI ĐI THI
# ==============================================================
"""
BẢNG SO SÁNH QUYẾT ĐỊNH ĐẦU RA (OUTPUT DECISION RULE):

1. MLE (Maximum Likelihood Estimation):
   - Đầu vào: alpha = 0
   - Điểm số: Score(c) = joint_likelihood(..., c, alpha=0)
   - Quyết định: best = argmax(Score)

2. MAP (Maximum A Posteriori):
   - Điểm số: Score(c) = prior(..., c) * joint_likelihood(..., c, alpha=alpha)
   - Hậu nghiệm chuẩn hóa: Posterior(c) = Score(c) / sum(Scores)
   - Quyết định: best = argmax(Posterior)

3. NBC (Naive Bayes Classifier):
   - Tử số(c) = prior(..., c) * joint_likelihood(..., c, alpha=alpha)
   - Mẫu số   = joint_evidence(..., x)
   - Kết quả theo slide: P_slide(c) = Tử số(c) / Mẫu số
   - Chuẩn hóa: P_norm(c) = Tử số(c) / sum(Tử số)
   - Quyết định: best = argmax(Tử số)
"""


# ==============================================================
# 9. VÍ DỤ THỰC THI KIỂM CHỨNG TẠI CHỖ
# ==============================================================
if __name__ == "__main__":
    df, target, attrs, values = load_data("data/datatemplate.csv")
    x_test = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
    
    print("===== KIỂM CHỨNG CÁC HÀM CƠ BẢN DÙNG CHUNG =====")
    print(f"Target: {target}")
    print(f"Attributes: {attrs}")
    print(f"Classes: {list(df[target].unique())}\n")
    
    for c in df[target].unique():
        p_c = prior(df, target, c)
        lik_prod = joint_likelihood(df, target, values, attrs, x_test, c, alpha=0)
        print(f"Lớp {c}: P({c}) = {p_c:.4f} | Π P(x_i|{c}) = {lik_prod:.6f} | Joint = {p_c * lik_prod:.6f}")
