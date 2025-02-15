import os
import csv
import json


def process_csv_files(input_folder, output_file):
    student_ids = []  # 用于存储所有学生编号

    # 遍历文件夹中的所有CSV文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                lines = list(csvreader)

                # 每三行作为一个整体处理，第一行为学生编号
                for i in range(0, len(lines), 3):
                    student_id = lines[i][0]  # 读取学生编号
                    if student_id == '0':
                        print('你好',filename,i)
                    else:
                        print('')
                    student_ids.append(student_id)

    # 删除重复的学生编号，并按升序排序
    student_ids = sorted(set(student_ids), key=int)

    # 创建从1开始的索引字典
    student_dict = {old_id: idx + 1 for idx, old_id in enumerate(student_ids)}

    # 保存为json文件
    with open(output_file, 'w') as jsonfile:
        json.dump(student_dict, jsonfile, ensure_ascii=False, indent=4)


# 使用示例
input_folder = './'  # 替换为实际CSV文件夹路径
output_file = 'user.json'  # 替换为实际输出文件路径
process_csv_files(input_folder, output_file)
