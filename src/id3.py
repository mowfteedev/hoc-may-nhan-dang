import os
import math
import pandas as pd

from rich.tree import Tree
from rich import print as rprint

# 1. Đọc dữ liệu
DATA_FILE = "data/data.csv" if os.path.exists("data/data.csv") else os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
df = pd.read_csv(DATA_FILE)
target = df.columns[-1]
attrs = list(df.columns[:-1])

D_test = {"Tiet_troi": "Nang", "Nhiet_do": "Lanh", "Do_am": "Cao", "Gio": "Manh"}


# 2. Các hàm cốt lõi
def entropy(D):
    p = D[target].value_counts(normalize=True)
    return -sum(x * math.log2(x) for x in p if x > 0)


def gain(D, a):
    w = D[a].value_counts(normalize=True)
    e = 0
    for v, sub in D.groupby(a):
        e += w[v] * entropy(sub)
    return entropy(D) - e


def odt(D, attrs):
    order = sorted(attrs, key=lambda a: gain(D, a), reverse=True)
    root = Tree(f"[bold cyan]{order[0]}[/bold cyan]")

    def build(sub_D, node, remaining, is_root=True):
        attr = remaining[0]
        rest = remaining[1:]
        for val, grp in sub_D.groupby(attr):
            prefix = f"{val}" if is_root else f"{attr} = {val}"
            if grp[target].nunique() == 1 or not rest:
                lbl = grp[target].iloc[0] if grp[target].nunique() == 1 else grp[target].mode()[0]
                color = "bold green" if lbl == "Co" else "bold red"
                node.add(f"[yellow]{prefix}[/yellow] -> [{color}]{lbl}[/{color}]")
            else:
                branch = node.add(f"[yellow]{prefix}[/yellow]")
                build(grp, branch, rest, is_root=False)

    build(D, root, order)
    rprint(root)
    return order


odt(df, attrs)

