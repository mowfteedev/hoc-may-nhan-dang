import math
import pandas as pd

# ---------- Đọc & Tiền xử lý dữ liệu ----------
DATA_FILE = "data/datatemplate.csv"
df = pd.read_csv(DATA_FILE).dropna()   # Đọc bảng và loại bỏ dòng thiếu dữ liệu
# df = df.drop(columns=["ID"])          # Bỏ cột ID/STT nếu có

target = df.columns[-1]                 # Cột cuối cùng là nhãn cần dự đoán (Class/Label)
attrs = list(df.columns[:-1])           # Danh sách các thuộc tính quan sát
values = {a: df[a].unique() for a in attrs}   # Mọi giá trị có thể của từng thuộc tính (tránh khuyết nhánh)


# ==============================================================
# BƯỚC 1: TÍNH ENTROPY H(D)
# Đo độ hỗn loạn thông tin của tập dữ liệu D
# ==============================================================
def entropy(D):
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p if x > 0)


# ==============================================================
# BƯỚC 2: TÍNH INFORMATION GAIN Gain(D, a)
# Đo độ lợi thông tin khi phân nhánh theo thuộc tính a
# ==============================================================
def gain(D, a):
    g = entropy(D)
    for v, sub in D.groupby(a):
        g -= len(sub) / len(D) * entropy(sub)
    return g


# ==============================================================
# BƯỚC 3: ĐƯA RA QUYẾT ĐỊNH (THUẬT TOÁN ID3)
# - Thuần nhất: Quyết định chọn nhãn lá (Yes/No)
# - Chưa thuần nhất: Quyết định chọn thuộc tính có Gain lớn nhất
# ==============================================================
def id3(D, current_attrs):
    major = D[target].mode()[0]                       # Nhãn đa số phòng ngừa nhánh rỗng

    # 1. Điều kiện dừng: Thuần nhất (Entropy = 0) hoặc đã hết thuộc tính chia
    if D[target].nunique() == 1 or not current_attrs:
        return major

    # 2. Tính Gain cho từng thuộc tính còn chia được
    gains = {a: gain(D, a) for a in current_attrs if D[a].nunique() > 1}
    if not gains:
        return major

    # 3. Quyết định: Chọn thuộc tính có Gain lớn nhất làm nút chia
    best = max(gains, key=gains.get)
    print(f"Chọn thuộc tính: {best} (Gain = {gains[best]:.4f})")

    # Phân nhánh đệ quy theo từng giá trị của thuộc tính tốt nhất
    rest = [a for a in current_attrs if a != best]
    children = {}
    for v in values[best]:
        sub = D[D[best] == v]
        children[v] = id3(sub, rest) if len(sub) else major
    return {"attr": best, "major": major, "children": children}


# ---------- Tiện ích: In cây quyết định trực quan ----------
def show(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "=> " + str(tree))
        return
    for v, sub in tree["children"].items():
        print(f"{indent}{tree['attr']} = {v}")
        show(sub, indent + "    ")


# ---------- Tiện ích: Dự đoán nhãn cho mẫu mới D_sample ----------
def predict(tree, D_sample):
    while isinstance(tree, dict):
        tree = tree["children"].get(D_sample[tree["attr"]], tree["major"])
    return tree


# ---------- Thực thi & Đánh giá ----------
tree = id3(df, attrs)

print("\n===== CÂY QUYẾT ĐỊNH =====")
show(tree)

# Đánh giá độ chính xác trên tập huấn luyện
pred = df.apply(lambda r: predict(tree, r), axis=1)
print(f"\nĐộ chính xác (train): {(pred == df[target]).mean():.2%}")

# Dự đoán mẫu quan sát mới D_mau
D_mau = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print(f"Dự đoán cho mẫu D_mau: {predict(tree, D_mau)}")
