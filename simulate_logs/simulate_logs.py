import pandas as pd
import time
import argparse
import os
from datetime import datetime, timedelta


def clean_field(value):
    """清理字段中的换行符和特殊字符"""
    if isinstance(value, str):
        return value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return value


def simulate_logs(input_file="../data/train.tsv", output_file="../data/simulate_log.tsv", sorted_file="../data/sorted_by_end.tsv",
                  start_time=None, end_time=None, accelerate=1, no_sleep=False):
    """
    模拟从 PENS 印象日志生成实时新闻点击日志流，按 end_time 排序。

    参数：
        input_file (str): 输入 TSV 文件路径，默认为 '../train.tsv'
        output_file (str): 输出 TSV 文件路径，默认为 'simulate_log.tsv'
        sorted_file (str): 按 end_time 排序的 TSV 文件路径，默认为 'sorted_by_end.tsv'
        start_time (str): 模拟开始时间（格式 YYYY-MM-DD HH:MM:SS），默认为最早 end_time
        end_time (str): 模拟结束时间（格式 YYYY-MM-DD HH:MM:SS），默认为最晚 end_time
        accelerate (float): 时间流速（每秒模拟时间增加秒数），默认为 1
        no_sleep (bool): 是否禁用延迟（快速测试），默认为 False
    """
    try:
        # 检查排序文件是否存在
        if os.path.exists(sorted_file):
            print(f"检测到排序文件 '{sorted_file}'，直接读取...")
            df = pd.read_csv(sorted_file, sep='\t', encoding='utf-8')
        else:
            print(f"排序文件 '{sorted_file}' 不存在，从 '{input_file}' 生成...")
            df = pd.read_csv(input_file, sep='\t', encoding='utf-8')

            # 验证必需字段
            required_columns = ['start', 'end']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"输入 TSV 文件必须包含以下列：{required_columns}")

            # 清理字段中的换行符
            print("清理字段中的换行符...")
            for col in df.columns:
                df[col] = df[col].apply(clean_field)

            # 解析 end 时间戳并排序
            print("解析 end 时间戳并排序...")
            df['end_timestamp'] = pd.to_datetime(df['end'], format='%m/%d/%Y %I:%M:%S %p')
            df = df.sort_values('end_timestamp')

            # 保存排序后的文件
            print(f"保存排序后的文件到 '{sorted_file}'...")
            df.drop(columns=['end_timestamp']).to_csv(sorted_file, sep='\t', index=False, encoding='utf-8',
                                                      lineterminator='\n')

        # 验证 end 时间戳
        if 'end_timestamp' not in df.columns:
            df['end_timestamp'] = pd.to_datetime(df['end'], format='%m/%d/%Y %I:%M:%S %p')

        # 过滤时间范围
        if start_time:
            try:
                start_time = pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S')
                df = df[df['end_timestamp'] >= start_time]
            except ValueError:
                raise ValueError("start_time 格式错误，需为 YYYY-MM-DD HH:MM:SS")
        if end_time:
            try:
                end_time = pd.to_datetime(end_time, format='%Y-%m-%d %H:%M:%S')
                df = df[df['end_timestamp'] <= end_time]
            except ValueError:
                raise ValueError("end_time 格式错误，需为 YYYY-MM-DD HH:MM:SS")

        if df.empty:
            raise ValueError("指定时间范围内的数据为空。")

        # 初始化模拟时间
        sim_start_time = start_time if start_time else df['end_timestamp'].iloc[0]
        sim_current_time = sim_start_time
        sim_end_time = df['end_timestamp'].iloc[-1] if end_time is None else min(pd.to_datetime(end_time),
                                                                                 df['end_timestamp'].iloc[-1])

        print(f"模拟时间跨度：{sim_start_time} 至 {sim_end_time}")
        print(f"时间流速：{accelerate} 秒/秒")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        # 初始化输出文件
        print(f"初始化输出文件 '{output_file}'...")
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            # 写入表头
            columns = [col for col in df.columns if col != 'end_timestamp']
            f.write('\t'.join(columns) + '\n')
            f.flush()

        # 模拟点击日志流
        print("开始模拟点击日志流...")
        processed_rows = 0

        start_real_time = time.time()

        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            while sim_current_time <= sim_end_time and not df.empty:
                # 查找 end_time <= 当前模拟时间的行
                rows_to_write = df[df['end_timestamp'] <= sim_current_time]

                if not rows_to_write.empty:
                    for _, row in rows_to_write.iterrows():
                        values = [str(row[col]) for col in df.columns if col != 'end_timestamp']
                        f.write('\t'.join(values) + '\n')
                        f.flush()
                        processed_rows += 1
                        if processed_rows % 1000 == 0:
                            print(f"已处理 {processed_rows} 行，当前模拟时间：{sim_current_time}")

                    # 移除已处理的行
                    df = df[df['end_timestamp'] > sim_current_time]

                # 更新模拟时间
                if not no_sleep:
                    time.sleep(1)  # 每秒更新一次

                sim_current_time = sim_end_time if (sim_current_time + timedelta(seconds=accelerate)) > sim_end_time else sim_current_time + timedelta(seconds=accelerate)

        print(f"模拟完成，已生成 '{output_file}'，总计处理 {processed_rows} 行")
        end_real_time = time.time()
        print(f"程序执行时间：{end_real_time-start_real_time}")

    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 未找到")
    except pd.errors.EmptyDataError:
        print(f"错误：文件 '{input_file}' 为空或格式不正确")
    except ValueError as e:
        print(f"错误：{str(e)}")
    except Exception as e:
        print(f"发生错误：{str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从 PENS 印象日志模拟实时新闻点击日志流，按 end_time 排序。')
    parser.add_argument('--input', default="../data/train.tsv", help='输入 TSV 文件路径')
    parser.add_argument('--output', default="../data/simulate_log.tsv", help='输出 TSV 文件路径')
    parser.add_argument('--sorted_file', default="../data/sorted_by_end.tsv",
                        help='按 end_time 排序的 TSV 文件路径')
    parser.add_argument('--start_time', default='2019-06-14 17:12:20', help='开始时间（格式 YYYY-MM-DD HH:MM:SS）')
    parser.add_argument('--end_time', default='2019-07-05 00:00:00', help='结束时间（格式 YYYY-MM-DD HH:MM:SS）')
    parser.add_argument('--accelerate', type=float, default=2400, help='时间流速（每秒模拟时间增加秒数，默认 1）')
    parser.add_argument('--no-sleep', action='store_true', help='禁用延迟，快速测试')
    args = parser.parse_args()

    simulate_logs(args.input, args.output, args.sorted_file, args.start_time, args.end_time, args.accelerate,
                  args.no_sleep)