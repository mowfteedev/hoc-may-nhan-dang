"""
DANH SÁCH CÁC HÀM TÍNH TOÁN CƠ BẢN DÙNG CHUNG CHO CÁC THUẬT TOÁN HỌC MÁY
========================================================================
Tên hàm sử dụng trực tiếp KÝ HIỆU TOÁN THỰC TẾ trên slide/bài giảng:

1. Xác suất & Thống kê Bayes (MLE, MAP, NBC):
   - P_h(df, target, h)                  : Tính xác suất tiên nghiệm P(h)
   - P_di_h(df, target, values, a, v, h) : Tính khả năng có điều kiện P(d_i | h)
   - P_D_h(df, target, values, attrs, D, h): Tích khả năng mẫu quan sát P(D | h) = Π P(d_i | h)
   - P_di(df, a, v)                      : Tính xác suất riêng/biên P(d_i)
   - P_D(df, attrs, D)                   : Mẫu số Naive Bayes P(D) = Π P(d_i)

2. Cây quyết định (ID3):
   - H_D(D, target)                      : Tính Entropy H(D)
   - Gain(D, target, a)                  : Tính Information Gain Gain(D, a)

3. Đánh giá kiểm thử:
   - accuracy(predict_fn, df, target)    : Tính tỷ lệ dự đoán chính xác trên dữ liệu

========================================================================
MA TRẬN HÀM VÀ THUẬT TOÁN SỬ DỤNG:
+-----------------------+-----+-----+-----+-----+
| Ký Hiệu Hàm           | ID3 | MLE | MAP | NBC |
+-----------------------+-----+-----+-----+-----+
| P_h()                 |     |     |  V  |  V  |
| P_di_h()              |     |  V  |  V  |  V  |
| P_D_h()               |     |  V  |  V  |  V  |
| P_di()                |     |     |     |  V  |
| P_D()                 |     |     |     |  V  |
| H_D()                 |  V  |     |     |     |
| Gain()                |  V  |     |     |     |
| accuracy()            |  V  |  V  |  V  |  V  |
+-----------------------+-----+-----+-----+-----+
========================================================================
"""

import math
import pandas as pd


# ==============================================================
# HÀM 1: P_h -> Tính P(h)
# [Ý NGHĨA]: Xác suất tiên nghiệm của giả thuyết h
# [THUẬT TOÁN SỬ DỤNG]: MAP, NBC
# ==============================================================
def P_h(df, target, h):
    """
    Tính P(h) = số mẫu thuộc giả thuyết h / tổng số mẫu
    - MAP: Score(h) = P(h) * P(D | h)
    - NBC: Tử số(h) = P(h) * P(D | h)
    - MLE: Không dùng (P(h) xem như bằng nhau cho mọi h).
    """
    return (df[target] == h).mean()


# ==============================================================
# HÀM 2: P_di_h -> Tính P(d_i | h)
# [Ý NGHĨA]: Xác suất thuộc tính a nhận giá trị v khi biết giả thuyết h
# [THUẬT TOÁN SỬ DỤNG]: MLE, MAP, NBC
# ==============================================================
def P_di_h(df, target, values, a, v, h, alpha=0):
    """
    Tính P(d_i | h) = P(a = v | h)
    Công thức tổng quát với làm trơn Laplace:
        P(d_i | h) = (số mẫu h có a=v + alpha) / (tổng mẫu h + alpha * |V_a|)

    - MLE: alpha = 0 -> count(a=v trong h) / count(h)
    - MAP / NBC: alpha = 0 (tính tay như slide) hoặc alpha = 1 (làm trơn Laplace)
    """
    sub = df[df[target] == h]
    k = len(values[a])
    count = (sub[a] == v).sum()
    return (count + alpha) / (len(sub) + alpha * k)


# ==============================================================
# HÀM 3: P_D_h -> Tính P(D | h) = Π P(d_i | h)
# [Ý NGHĨA]: Khả năng (Likelihood) của toàn bộ mẫu D khi biết h
# [THUẬT TOÁN SỬ DỤNG]: MLE, MAP, NBC (Bước lõi của cả 3 thuật toán)
# ==============================================================
def P_D_h(df, target, values, attrs, D, h, alpha=0, verbose=False):
    """
    Tính tích các P(d_i | h) của toàn bộ mẫu quan sát D:
        P(D | h) = Π P(a_i = D[a_i] | h)

    - MLE: Đây CHÍNH LÀ ĐẦU RA mục tiêu: L(h) = P(D | h) với alpha=0
           => h_ML = argmax P(D | h)
    - MAP: Nhân với P_h(h) để tính hậu nghiệm:
           Score(h) = P(h) * P(D | h)
           => h_MAP = argmax [P(h) * P(D | h)]
    - NBC: Nhân với P_h(h) để làm Tử số của NBC
    """
    prod = 1.0
    for a in attrs:
        p = P_di_h(df, target, values, a, D[a], h, alpha)
        if verbose:
            print(f"   P({a}={D[a]} | {h}) = {p:.4f}")
        prod *= p
    return prod


# ==============================================================
# HÀM 4: P_di -> Tính P(d_i)
# [Ý NGHĨA]: Xác suất riêng của thuộc tính a nhận giá trị v
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất NBC
# ==============================================================
def P_di(df, a, v):
    """
    Tính P(d_i) = P(a = v) trên toàn bộ bảng dữ liệu
    - Dùng làm nhân tử trong mẫu số Naive Bayes theo slide: Π P(d_i)
    """
    return (df[a] == v).mean()


# ==============================================================
# HÀM 5: P_D -> Tính P(D) = Π P(d_i)
# [Ý NGHĨA]: Mẫu số Naive Bayes (xác suất biên của mẫu quan sát D)
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất NBC
# ==============================================================
def P_D(df, attrs, D, verbose=False):
    """
    Tính mẫu số theo slide của Naive Bayes:
        P(D) = Π P(a_i = D[a_i])

    Khi đó theo slide:
        P_slide(h | D) = [P(h) * P(D | h)] / P(D)
    """
    prod = 1.0
    for a in attrs:
        p = P_di(df, a, D[a])
        if verbose:
            print(f"P({a}={D[a]}) = {p:.4f}")
        prod *= p
    return prod


# ==============================================================
# HÀM 6: H_D -> Tính H(D)
# [Ý NGHĨA]: Entropy (độ hỗn loạn thông tin) của tập dữ liệu D
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất ID3
# ==============================================================
def H_D(D, target):
    """
    Tính Entropy H(D):
        H(D) = - Σ [ p_h * log2(p_h) ]
    """
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p if x > 0)


# ==============================================================
# HÀM 7: Gain -> Tính Gain(D, a)
# [Ý NGHĨA]: Độ lợi thông tin khi phân nhánh theo thuộc tính a
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất ID3
# ==============================================================
def Gain(D, target, a):
    """
    Tính Information Gain:
        Gain(D, a) = H(D) - Σ [ (|D_v| / |D|) * H(D_v) ]

    - ID3 chọn thuộc tính a có Gain(D, a) lớn nhất làm nút rẽ.
    """
    g = H_D(D, target)
    for _, sub in D.groupby(a):
        g -= len(sub) / len(D) * H_D(sub, target)
    return g


# ==============================================================
# HÀM 8: accuracy
# [Ý NGHĨA]: Tính tỷ lệ phân loại chính xác trên tập dữ liệu
# [THUẬT TOÁN SỬ DỤNG]: ID3, MLE, MAP, NBC (Toàn bộ)
# ==============================================================
def accuracy(predict_fn, df, target):
    """
    Tính Accuracy = số mẫu dự đoán đúng / tổng số mẫu
    - predict_fn: hàm nhận vào 1 dòng dữ liệu và trả về nhãn dự đoán h
    """
    def _extract(row):
        res = predict_fn(row)
        return res[0] if isinstance(res, (tuple, list)) else res

    pred = df.apply(_extract, axis=1)
    return (pred == df[target]).mean()


# ==============================================================
# ALIASES (Bí danh tương đương để tương thích tên gọi tiếng Anh)
# ==============================================================
prior = P_h
likelihood = P_di_h
joint_likelihood = P_D_h
evidence = P_di
joint_evidence = P_D
entropy = H_D
gain = Gain
evaluate_accuracy = accuracy
