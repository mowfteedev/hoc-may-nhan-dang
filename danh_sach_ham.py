"""
DANH SÁCH CÁC HÀM TÍNH TOÁN CƠ BẢN DÙNG CHUNG CHO CÁC THUẬT TOÁN HỌC MÁY
========================================================================
Quy ước ký hiệu chuẩn hóa toàn bộ dự án:
- D     : Mẫu dữ liệu quan sát cần phân loại (dạng dict: {"a1": v1, ...})
- h     : Giả thuyết / Lớp mục tiêu cần dự đoán (h in H)
- df    : DataFrame dữ liệu
- target: Tên cột nhãn mục tiêu (cột cuối cùng)
- attrs : Danh sách tên các thuộc tính đặc trưng [a1, a2, ...]
- values: Dict tập các giá trị có thể của từng thuộc tính {a: [v1, v2, ...]}
- alpha : Tham số làm trơn Laplace (0: tính tay/MLE; 1: làm trơn Laplace)
- a     : Thuộc tính đặc trưng (Attribute)
- v     : Giá trị của thuộc tính (Value)

MA TRẬN HÀM VÀ THUẬT TOÁN ÁP DỤNG:
+-----------------------+-----+-----+-----+-----+
| Tên Hàm               | ID3 | MLE | MAP | NBC |
+-----------------------+-----+-----+-----+-----+
| prior()               |     |     |  V  |  V  |
| likelihood()          |     |  V  |  V  |  V  |
| joint_likelihood()    |     |  V  |  V  |  V  |
| evidence()            |     |     |     |  V  |
| joint_evidence()      |     |     |     |  V  |
| entropy()             |  V  |     |     |     |
| gain()                |  V  |     |     |     |
| evaluate_accuracy()   |  V  |  V  |  V  |  V  |
+-----------------------+-----+-----+-----+-----+
========================================================================
"""

import math
import pandas as pd


# ==============================================================
# HÀM 1: prior
# [KÝ HIỆU TOÁN]: P(h)
# [THUẬT TOÁN SỬ DỤNG]: MAP, NBC
# ==============================================================
def prior(df, target, h):
    """
    Tính xác suất tiên nghiệm P(h) của giả thuyết h.

    [CÁCH DÙNG]:
    - MAP: Score(h) = P(h) * P(D | h)
    - NBC: Tử số(h) = P(h) * P(D | h)
    - MLE: Không dùng (giả định phân phối đều, P(h) như nhau cho mọi h).
    """
    return (df[target] == h).mean()


# ==============================================================
# HÀM 2: likelihood
# [KÝ HIỆU TOÁN]: P(a = v | h) hoặc P(d_i | h)
# [THUẬT TOÁN SỬ DỤNG]: MLE, MAP, NBC
# ==============================================================
def likelihood(df, target, values, a, v, h, alpha=0):
    """
    Tính khả năng P(a = v | h) khi biết giả thuyết h.

    [CÁCH DÙNG]:
    - MLE: alpha = 0 -> P = count(a=v trong h) / count(h)
    - MAP: alpha = 0 (tính tay như slide) hoặc alpha = 1 (Laplace)
    - NBC: alpha = 0 (tính tay như slide) hoặc alpha = 1 (Laplace)
    """
    sub = df[df[target] == h]
    k = len(values[a])
    count = (sub[a] == v).sum()
    return (count + alpha) / (len(sub) + alpha * k)


# ==============================================================
# HÀM 3: joint_likelihood
# [KÝ HIỆU TOÁN]: P(D | h) = Π P(d_i | h)
# [THUẬT TOÁN SỬ DỤNG]: MLE, MAP, NBC
# ==============================================================
def joint_likelihood(df, target, values, attrs, D, h, alpha=0, verbose=False):
    """
    Tính tích các khả năng của toàn bộ mẫu quan sát D khi biết giả thuyết h.

    [CÁCH DÙNG]:
    - MLE: Đây chính là hàm mục tiêu: L(h) = P(D | h) với alpha=0.
           Quyết định MLE: h_ML = argmax L(h).
    - MAP: Nhân với prior(h) để tính hậu nghiệm: Score(h) = P(h) * P(D | h).
           Quyết định MAP: h_MAP = argmax Score(h).
    - NBC: Nhân với prior(h) để tính Tử số của NBC.
    """
    prod = 1.0
    for a in attrs:
        p = likelihood(df, target, values, a, D[a], h, alpha)
        if verbose:
            print(f"   P({a}={D[a]} | {h}) = {p:.4f}")
        prod *= p
    return prod


# ==============================================================
# HÀM 4: evidence
# [KÝ HIỆU TOÁN]: P(a = v) hoặc P(d_i)
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất NBC
# ==============================================================
def evidence(df, a, v):
    """
    Tính xác suất biên P(a = v) trên toàn bộ dữ liệu.

    [CÁCH DÙNG]:
    - NBC: Làm nhân tử trong mẫu số Naive Bayes theo slide: Π P(d_i).
    """
    return (df[a] == v).mean()


# ==============================================================
# HÀM 5: joint_evidence
# [KÝ HIỆU TOÁN]: P(D) = Π P(d_i)
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất NBC
# ==============================================================
def joint_evidence(df, attrs, D, verbose=False):
    """
    Tính tích xác suất biên của mẫu quan sát D (Mẫu số Naive Bayes).

    [CÁCH DÙNG]:
    - NBC: Mẫu số = Π P(d_i).
           P_slide(h | D) = [P(h) * Π P(d_i | h)] / [Π P(d_i)]
    """
    prod = 1.0
    for a in attrs:
        p = evidence(df, a, D[a])
        if verbose:
            print(f"P({a}={D[a]}) = {p:.4f}")
        prod *= p
    return prod


# ==============================================================
# HÀM 6: entropy
# [KÝ HIỆU TOÁN]: H(D)
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất ID3
# ==============================================================
def entropy(D, target):
    """
    Tính Entropy (độ hỗn loạn thông tin) của tập dữ liệu D.

    [CÁCH DÙNG]:
    - ID3: H(D) = - Σ [ p_h * log2(p_h) ]
    """
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p if x > 0)


# ==============================================================
# HÀM 7: gain
# [KÝ HIỆU TOÁN]: Gain(D, a)
# [THUẬT TOÁN SỬ DỤNG]: Duy nhất ID3
# ==============================================================
def gain(D, target, a):
    """
    Tính độ lợi thông tin Gain(D, a) khi chọn thuộc tính a để phân nhánh.

    [CÁCH DÙNG]:
    - ID3: Gain(D, a) = Entropy(D) - Σ [ (|D_v| / |D|) * Entropy(D_v) ]
           Chọn thuộc tính có Gain lớn nhất làm gốc nhánh.
    """
    g = entropy(D, target)
    for _, sub in D.groupby(a):
        g -= len(sub) / len(D) * entropy(sub, target)
    return g


# ==============================================================
# HÀM 8: evaluate_accuracy
# [THUẬT TOÁN SỬ DỤNG]: ID3, MLE, MAP, NBC (Toàn bộ)
# ==============================================================
def evaluate_accuracy(predict_fn, df, target):
    """
    Đánh giá độ chính xác (Accuracy) của hàm phân loại predict_fn trên df.

    [CÁCH DÙNG]:
    - Áp dụng cho mọi thuật toán sau khi đã huấn luyện xong.
    - predict_fn(row) trả về nhãn dự đoán h_pred (hoặc tuple (h_pred, ...)).
    """
    def _extract_pred(row):
        res = predict_fn(row)
        return res[0] if isinstance(res, (tuple, list)) else res

    pred = df.apply(_extract_pred, axis=1)
    return (pred == df[target]).mean()
