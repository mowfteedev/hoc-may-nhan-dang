"""
DANH SÁCH CÁC HÀM CƠ BẢN DÙNG CHUNG CHO CÁC THUẬT TOÁN HỌC MÁY
================================================================
Mục đích:
- Chuẩn hóa các bước tính toán giống nhau giữa các bài toán thi thực hành.
- Mỗi hàm đều có CHÚ THÍCH RÕ RÀNG các thuật toán sẽ sử dụng và cách dùng cụ thể.
- Phần mở rộng riêng biệt của từng thuật toán sẽ được triển khai trong thư mục code/.

MA TRẬN HÀM VÀ THUẬT TOÁN (CHEATSHEET PHÒNG THI):
+-----------------------+-----+-----+-----+-----+
| Tên Hàm               | ID3 | MLE | MAP | NBC |
+-----------------------+-----+-----+-----+-----+
| load_data()           |  V  |  V  |  V  |  V  |
| prior()               |     |     |  V  |  V  |
| likelihood()          |     |  V  |  V  |  V  |
| joint_likelihood()    |     |  V  |  V  |  V  |
| evidence()            |     |     |     |  V  |
| joint_evidence()      |     |     |     |  V  |
| entropy()             |  V  |     |     |     |
| gain()                |  V  |     |     |     |
| evaluate_accuracy()   |  V  |  V  |  V  |  V  |
+-----------------------+-----+-----+-----+-----+
================================================================
"""

import math
import pandas as pd


# ==============================================================
# HÀM 1: load_data
# [THUẬT TOÁN SỬ DỤNG]: ID3, MLE, MAP, NBC (Tất cả bài toán)
# [MỤC ĐÍCH]: Đọc CSV và tiền xử lý cấu trúc dữ liệu chuẩn
# ==============================================================
def load_data(filepath="data/datatemplate.csv"):
    """
    [THUẬT TOÁN SỬ DỤNG]: ID3, MLE, MAP, NBC
    [CÁCH DÙNG]:
    - ID3 : Cần cả df, target, attrs, values (để phân nhánh cây không khuyết).
    - MLE : Cần df, target, attrs.
    - MAP : Cần df, target, attrs, values (để tính k làm trơn Laplace).
    - NBC : Cần df, target, attrs, values (để tính k làm trơn Laplace).
    """
    df = pd.read_csv(filepath).dropna()
    # df = df.drop(columns=["ID"])  # Bỏ cột ID/STT nếu đề bài có

    target = df.columns[-1]                 # Cột cuối cùng là nhãn phân lớp (Class/Label)
    attrs = list(df.columns[:-1])           # Danh sách các thuộc tính quan sát
    values = {a: df[a].unique() for a in attrs}  # Mọi giá trị có thể của từng thuộc tính
    return df, target, attrs, values


# ==============================================================
# HÀM 2: prior
# [THUẬT TOÁN SỬ DỤNG]: MAP, NBC
# [MỤC ĐÍCH]: Tính xác suất tiên nghiệm P(c) của lớp c
# ==============================================================
def prior(df, target, c):
    """
    [THUẬT TOÁN SỬ DỤNG]: MAP, NBC
    [CÁCH DÙNG]:
    - MAP: Nhân vào điểm số giả thuyết: Score(c) = P(c) * Π P(d_i | c).
    - NBC: Nhân vào tử số Bayes: Tử số(c) = P(c) * Π P(x_i | c).
    - MLE: KHÔNG DÙNG (MLE giả định phân phối đều, xem như P(c) = 1.0 cho mọi lớp).
    - ID3: KHÔNG DÙNG.
    """
    return (df[target] == c).mean()


# ==============================================================
# HÀM 3: likelihood
# [THUẬT TOÁN SỬ DỤNG]: MLE, MAP, NBC
# [MỤC ĐÍCH]: Tính khả năng có điều kiện P(a = v | c)
# ==============================================================
def likelihood(df, target, values, a, v, c, alpha=0):
    """
    [THUẬT TOÁN SỬ DỤNG]: MLE, MAP, NBC
    [CÁCH DÙNG]:
    - MLE: Truyền alpha = 0.
           Công thức: P = (số mẫu lớp c có a = v) / (tổng số mẫu lớp c)
    - MAP: Truyền alpha = 0 (tính tay như slide) hoặc alpha = 1 (làm trơn Laplace).
           Công thức: P = (số mẫu + alpha) / (tổng mẫu + alpha * |V_a|)
    - NBC: Truyền alpha = 0 (tính tay như slide) hoặc alpha = 1 (làm trơn Laplace).
    - ID3: KHÔNG DÙNG.
    """
    sub = df[df[target] == c]
    k = len(values[a])
    count = (sub[a] == v).sum()
    return (count + alpha) / (len(sub) + alpha * k)


# ==============================================================
# HÀM 4: joint_likelihood
# [THUẬT TOÁN SỬ DỤNG]: MLE, MAP, NBC
# [MỤC ĐÍCH]: Tích các khả năng của mẫu quan sát x với lớp c
#             Π P(a_i = x[a_i] | c)
# ==============================================================
def joint_likelihood(df, target, values, attrs, x, c, alpha=0, verbose=False):
    """
    [THUẬT TOÁN SỬ DỤNG]: MLE, MAP, NBC
    [CÁCH DÙNG]:
    - MLE: Đây CHÍNH LÀ ĐẦU RA của MLE: L(c) = joint_likelihood(..., alpha=0).
           Quyết định MLE: c_ML = argmax(L).
    - MAP: Lấy kết quả này nhân với prior(c) để ra Score(c).
           Quyết định MAP: c_MAP = argmax(prior * joint_likelihood).
    - NBC: Lấy kết quả này nhân với prior(c) để ra Tử số của NBC.
    """
    prod = 1.0
    for a in attrs:
        p = likelihood(df, target, values, a, x[a], c, alpha)
        if verbose:
            print(f"   P({a}={x[a]} | {c}) = {p:.4f}")
        prod *= p
    return prod


# ==============================================================
# HÀM 5: evidence
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất NBC
# [MỤC ĐÍCH]: Tính xác suất biên P(a = v) của thuộc tính
# ==============================================================
def evidence(df, a, v):
    """
    [THUẬT TOÁN SỬ DỤNG]: Duy nhất NBC
    [CÁCH DÙNG]:
    - NBC: Tính xác suất riêng P(a = v) trên toàn bộ bảng dữ liệu.
           Dùng làm nhân tử trong mẫu số của công thức slide Naive Bayes.
    - MLE, MAP, ID3: KHÔNG DÙNG.
    """
    return (df[a] == v).mean()


# ==============================================================
# HÀM 6: joint_evidence
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất NBC
# [MỤC ĐÍCH]: Tính mẫu số của Naive Bayes theo slide: Π P(x_i)
# ==============================================================
def joint_evidence(df, attrs, x, verbose=False):
    """
    [THUẬT TOÁN SỬ DỤNG]: Duy nhất NBC
    [CÁCH DÙNG]:
    - NBC: Tính tích xác suất biên của tất cả thuộc tính trong mẫu x.
           Mẫu số = Π P(a_i = x[a_i]).
           Sau đó: P_slide(c) = Tử số(c) / Mẫu số.
    - MLE, MAP, ID3: KHÔNG DÙNG.
    """
    prod = 1.0
    for a in attrs:
        p = evidence(df, a, x[a])
        if verbose:
            print(f"P({a}={x[a]}) = {p:.4f}")
        prod *= p
    return prod


# ==============================================================
# HÀM 7: entropy
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất ID3
# [MỤC ĐÍCH]: Tính độ hỗn loạn thông tin H(S)
# ==============================================================
def entropy(d, target):
    """
    [THUẬT TOÁN SỬ DỤNG]: Duy nhất ID3
    [CÁCH DÙNG]:
    - ID3: Tính Entropy của tập con dữ liệu d:
           H(S) = - Σ [ p_i * log2(p_i) ]
    - MLE, MAP, NBC: KHÔNG DÙNG.
    """
    p = d[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p if x > 0)


# ==============================================================
# HÀM 8: gain
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất ID3
# [MỤC ĐÍCH]: Tính độ lợi thông tin Gain(S, A) để chọn nút chia
# ==============================================================
def gain(d, target, attr):
    """
    [THUẬT TOÁN SỬ DỤNG]: Duy nhất ID3
    [CÁCH DÙNG]:
    - ID3: Tính Information Gain khi chia tập d theo thuộc tính attr:
           Gain(S, A) = Entropy(S) - Σ [ (|S_v| / |S|) * Entropy(S_v) ]
           Nút có Gain lớn nhất sẽ được chọn làm gốc của cây con.
    - MLE, MAP, NBC: KHÔNG DÙNG.
    """
    g = entropy(d, target)
    for _, sub in d.groupby(attr):
        g -= len(sub) / len(d) * entropy(sub, target)
    return g


# ==============================================================
# HÀM 9: evaluate_accuracy
# [THUẬT TOÁN SỬ DỤNG]: ID3, MLE, MAP, NBC (Tất cả bài toán)
# [MỤC ĐÍCH]: Đánh giá độ chính xác phân lớp trên tập dữ liệu
# ==============================================================
def evaluate_accuracy(predict_fn, df, target):
    """
    [THUẬT TOÁN SỬ DỤNG]: ID3, MLE, MAP, NBC
    [CÁCH DÙNG]:
    - Nhận vào hàm dự đoán predict_fn(x) của bất kỳ thuật toán nào.
    - Trả về độ chính xác: số mẫu đúng / tổng số mẫu.
    - Ví dụ: acc = evaluate_accuracy(lambda x: id3_predict(tree, x), df, target)
    """
    def _extract_pred(row):
        res = predict_fn(row)
        return res[0] if isinstance(res, (tuple, list)) else res

    pred = df.apply(_extract_pred, axis=1)
    acc = (pred == df[target]).mean()
    return acc


# ==============================================================
# HƯỚNG DẪN MỞ RỘNG ĐẦU RA TRONG THƯ MỤC code/ KHI ĐI THI
# ==============================================================
"""
QUY TẮC QUYẾT ĐỊNH ĐẦU RA (DECISION RULE) CHO TỪNG BÀI TOÁN:

1. BÀI TOÁN MLE (code/mle/mle.py):
   Score(c) = joint_likelihood(df, target, values, attrs, x, c, alpha=0)
   Quyết định: c_ML = argmax(Score)

2. BÀI TOÁN MAP (code/map/map.py):
   Score(c) = prior(df, target, c) * joint_likelihood(df, target, values, attrs, x, c, alpha=alpha)
   Posterior(c) = Score(c) / sum(Score.values())
   Quyết định: c_MAP = argmax(Posterior)

3. BÀI TOÁN NBC (code/nbc/nbc.py):
   Tử số(c) = prior(df, target, c) * joint_likelihood(df, target, values, attrs, x, c, alpha=alpha)
   Mẫu số   = joint_evidence(df, attrs, x)
   P_theo_slide(c) = Tử số(c) / Mẫu số
   P_chuan_hoa(c)  = Tử số(c) / sum(Tử số.values())
   Quyết định: c_NBC = argmax(Tử số)

4. BÀI TOÁN ID3 (code/id3/id3.py):
   Tìm best_attr = argmax_{a} gain(d, target, a)
   Phân nhánh theo từng giá trị trong values[best_attr]
"""


# ==============================================================
# VÍ DỤ CHẠY THỬ NGHIỆM TẠI CHỖ
# ==============================================================
if __name__ == "__main__":
    df, target, attrs, values = load_data("data/datatemplate.csv")
    x_test = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}

    print("===== CHẠY THỬ HÀM DÙNG CHUNG TRÊN DỮ LIỆU THỜI TIẾT =====")
    print(f"Nhãn mục tiêu: {target}")
    print(f"Thuộc tính: {attrs}")
    print(f"Không gian lớp: {list(df[target].unique())}\n")

    print(f"Entropy tập gốc S: {entropy(df, target):.4f}")
    for a in attrs:
        print(f"Gain(S, {a}) = {gain(df, target, a):.4f}")

    print("\n--- Tính toán Bayes cho mẫu thử nghiệm ---")
    for c in df[target].unique():
        p_c = prior(df, target, c)
        lik = joint_likelihood(df, target, values, attrs, x_test, c, alpha=0)
        print(f"Lớp {c}: Prior = {p_c:.4f} | Tích Likelihood = {lik:.6f} | Joint = {p_c * lik:.6f}")
