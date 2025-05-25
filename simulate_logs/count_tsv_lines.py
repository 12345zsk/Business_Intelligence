import pandas as pd
import argparse
import os


def count_tsv_rows(input_file="../data/train.tsv"):
    """
    计算 TSV 文件的数据行数（不包括表头）。

    参数：
        input_file (str): 输入 TSV 文件路径，默认为 '../train.tsv'

    返回：
        None，打印行数到控制台
    """
    try:
        # 读取 TSV 文件
        df = pd.read_csv(input_file, sep='\t', encoding='utf-8')

        # 计算数据行数（不包括表头）
        row_count = len(df)

        # 检查是否为空
        if row_count == 0:
            print(f"文件 '{input_file}' 为空（无数据行）。")
            return

        # 打印行数
        print(f"文件 '{input_file}' 的数据行数：{row_count}")

    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 未找到")
    except pd.errors.EmptyDataError:
        print(f"错误：文件 '{input_file}' 为空或格式不正确")
    except Exception as e:
        print(f"发生错误：{str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算 TSV 文件的数据行数。')
    parser.add_argument('--input', default="../data/train.tsv", help='输入 TSV 文件路径')
    args = parser.parse_args()

    count_tsv_rows(args.input)