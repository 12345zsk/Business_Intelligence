import pandas as pd
import argparse
import os
from datetime import timedelta


def check_time_span(input_file):
    """
    查看 TSV 文件中 end 列的最大时间跨度。

    参数：
        input_file (str): 输入 TSV 文件路径
    """
    try:
        # 读取 TSV 文件
        print(f"正在读取文件 '{input_file}'...")
        df = pd.read_csv(input_file, sep='\t', encoding='utf-8')

        # 验证 end 列存在
        if 'end' not in df.columns:
            raise ValueError("TSV 文件必须包含 'end' 列")

        # 解析 end 列时间戳
        print("解析 end 列时间戳...")
        df['end_timestamp'] = pd.to_datetime(df['end'], format='%m/%d/%Y %I:%M:%S %p')

        # 计算最大和最小时间
        min_time = df['end_timestamp'].min()
        max_time = df['end_timestamp'].max()

        # 计算时间跨度
        time_span = max_time - min_time

        # 格式化为 HH:MM:SS
        time_span_formatted = str(timedelta(seconds=int(time_span.total_seconds())))

        # 输出结果
        print(f"最小 end 时间：{min_time}")
        print(f"最大 end 时间：{max_time}")
        print(f"最大时间跨度：{time_span_formatted}")

        return time_span

    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 未找到")
        return None
    except pd.errors.EmptyDataError:
        print(f"错误：文件 '{input_file}' 为空或格式不正确")
        return None
    except ValueError as e:
        print(f"错误：{str(e)}")
        return None
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='查看 TSV 文件中 end 列的最大时间跨度。')
    parser.add_argument('--input', default="../data/train.tsv", help='输入 TSV 文件路径')
    args = parser.parse_args()

    check_time_span(args.input)