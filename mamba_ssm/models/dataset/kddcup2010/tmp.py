import csv
# /home/shaomingxing/.virtualenvs/ATKT-main/bin/python /home/shaomingxing/fuxian/MambaTran/ATKT-main/dataset/kddcup2010/tmp.py
# The overall maximum value is: 660
# 读取CSV文件
file_path = 'kddcup2010_valid4.csv'

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    lines = list(reader)

# 定义需要查找最大值的行号，注意Python索引从0开始
rows_to_check = range(0, len(lines), 3)  # 第2行，第5行，第8行等
overall_max_value = float('-inf')  # 初始化为负无穷

# 遍历指定的行并计算最大值
for row in rows_to_check:
    if row < len(lines):
        row_max_value = max(map(int, lines[row]))
        if row_max_value > overall_max_value:
            overall_max_value = row_max_value

# 输出结果
print(f"The overall maximum value is: {overall_max_value}")
