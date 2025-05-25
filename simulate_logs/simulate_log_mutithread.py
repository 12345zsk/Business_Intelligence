import pandas as pd
import time
import argparse
import os
import threading
from datetime import datetime, timedelta


def clean_field(value):
    """清理字段中的换行符和特殊字符"""
    if isinstance(value, str):
        return value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return value


def process_thread(df, thread_id, sim_start_time, sim_end_time, accelerate, no_sleep, output_file):
    """线程处理函数，模拟子数据集的点击日志流，写入独立文件"""
    sim_current_time = sim_start_time
    processed_rows = 0
    thread_output = f"{output_file}_thread_{thread_id}.tsv"

    try:
        # 初始化线程输出文件
        with open(thread_output, 'w', newline='', encoding='utf-8') as f:
            columns = [col for col in df.columns if col != 'end_timestamp'] + ['thread_id']
            f.write('\t'.join(columns) + '\n')
            f.flush()

        while sim_current_time <= sim_end_time and not df.empty:
            # 查找 end_time <= 当前模拟时间的行
            rows_to_write = df[df['end_timestamp'] <= sim_current_time]

            if not rows_to_write.empty:
                with open(thread_output, 'a', newline='', encoding='utf-8') as f:
                    for _, row in rows_to_write.iterrows():
                        values = [str(row[col]) for col in df.columns if col != 'end_timestamp'] + [str(thread_id)]
                        f.write('\t'.join(values) + '\n')
                        f.flush()
                        processed_rows += 1
                        if processed_rows % 1000 == 0:
                            print(f"线程 {thread_id} 已处理 {processed_rows} 行，当前模拟时间：{sim_current_time}")

                # 移除已处理的行
                df = df[df['end_timestamp'] > sim_current_time]

            # 更新模拟时间
            if not no_sleep:
                time.sleep(1)  # 每秒更新一次
            next_time = sim_current_time + timedelta(seconds=accelerate)
            sim_current_time = min(next_time, sim_end_time)

    except Exception as e:
        print(f"线程 {thread_id} 发生错误：{str(e)}")

    print(f"线程 {thread_id} 完成，总计处理 {processed_rows} 行，输出到 '{thread_output}'")


def merge_thread_files(output_file, num_threads):
    """合并所有线程的临时文件，按 end 时间排序"""
    try:
        all_dfs = []
        temp_files = []

        # 读取每个线程的临时文件
        for i in range(num_threads):
            temp_file = f"{output_file}_thread_{i}.tsv"
            if os.path.exists(temp_file):
                temp_df = pd.read_csv(temp_file, sep='\t', encoding='utf-8')
                temp_df['end_timestamp'] = pd.to_datetime(temp_df['end'], format='%m/%d/%Y %I:%M:%S %p')
                all_dfs.append(temp_df)
                temp_files.append(temp_file)

        if not all_dfs:
            print("无临时文件可合并")
            return

        # 合并并排序
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df = merged_df.sort_values('end_timestamp')
        merged_df = merged_df.drop(columns=['end_timestamp'])

        # 写入最终输出文件
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            f.write('\t'.join(merged_df.columns) + '\n')
            merged_df.to_csv(f, sep='\t', index=False, header=False, lineterminator='\n')

        print(f"已合并 {len(temp_files)} 个临时文件到 '{output_file}'，总行数：{len(merged_df)}")

        # 删除临时文件
        for temp_file in temp_files:
            os.remove(temp_file)
            print(f"已删除临时文件 '{temp_file}'")

    except Exception as e:
        print(f"合并文件时发生错误：{str(e)}")


def simulate_logs(input_file="../data/train.tsv", output_file="../data/simulate_log.tsv", sorted_file="../data/sorted_by_end.tsv",
                  start_time=None, end_time=None, accelerate=1, no_sleep=False, num_threads=1):
    """
    模拟从 PENS 印象日志生成实时新闻点击日志流，按 end_time 排序，支持多线程分开写入。

    参数：
        input_file (str): 输入 TSV 文件路径，默认为 '../train.tsv'
        output_file (str): 输出 TSV 文件路径，默认为 'simulate_log.tsv'
        sorted_file (str): 按 end_time 排序的 TSV 文件路径，默认为 'sorted_by_end.tsv'
        start_time (str): 模拟开始时间（格式 YYYY-MM-DD HH:MM:SS），默认为最早 end_time
        end_time (str): 模拟结束时间（格式 YYYY-MM-DD HH:MM:SS），默认为最晚 end_time
        accelerate (float): 时间流速（每秒模拟时间增加秒数），默认为 1
        no_sleep (bool): 是否禁用延迟（快速测试），默认为 False
        num_threads (int): 线程数量，默认为 1
    """
    # 记录程序开始时间
    start_real_time = time.time()

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
        sim_end_time = df['end_timestamp'].iloc[-1] if end_time is None else min(pd.to_datetime(end_time),
                                                                                 df['end_timestamp'].iloc[-1])

        print(f"模拟时间跨度：{sim_start_time} 至 {sim_end_time}")
        print(f"时间流速：{accelerate} 秒/秒，线程数：{num_threads}")

        # 划分数据到线程
        if num_threads < 1:
            raise ValueError("线程数必须大于 0")

        threads = []
        time_range = (sim_end_time - sim_start_time).total_seconds() / num_threads
        for i in range(num_threads):
            thread_start = sim_start_time + timedelta(seconds=time_range * i)
            thread_end = sim_start_time + timedelta(
                seconds=time_range * (i + 1)) if i < num_threads - 1 else sim_end_time
            thread_df = df[(df['end_timestamp'] >= thread_start) & (df['end_timestamp'] <= thread_end)]

            if not thread_df.empty:
                thread = threading.Thread(
                    target=process_thread,
                    args=(thread_df, i, sim_start_time, sim_end_time, accelerate, no_sleep, output_file)
                )
                threads.append(thread)
                print(f"启动线程 {i}，处理时间范围：{thread_start} 至 {thread_end}")
            else:
                print(f"线程 {i} 无数据，跳过（时间范围：{thread_start} 至 {thread_end}）")

        # 启动并等待线程
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # 合并临时文件
        print("开始合并线程输出文件...")
        merge_thread_files(output_file, num_threads)

        # 计算并输出执行时间
        end_real_time = time.time()
        execution_time = end_real_time - start_real_time
        execution_time_formatted = str(timedelta(seconds=int(execution_time)))
        print(f"所有线程完成，已生成 '{output_file}'")
        print(f"程序执行时间：{execution_time:.2f} 秒（{execution_time_formatted}）")

    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 未找到")
    except pd.errors.EmptyDataError:
        print(f"错误：文件 '{input_file}' 为空或格式不正确")
    except ValueError as e:
        print(f"错误：{str(e)}")
    except Exception as e:
        print(f"发生错误：{str(e)}")
    finally:
        end_real_time = time.time()
        execution_time = end_real_time - start_real_time
        execution_time_formatted = str(timedelta(seconds=int(execution_time)))
        print(f"程序执行时间：{execution_time:.2f} 秒（{execution_time_formatted}）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='从 PENS 印象日志模拟实时新闻点击日志流，按 end_time 排序，支持多线程分开写入。')
    parser.add_argument('--input', default="../data/train.tsv", help='输入 TSV 文件路径')
    parser.add_argument('--output', default="../data/simulate_log_multi-thread.tsv", help='输出 TSV 文件路径')
    parser.add_argument('--sorted_file', default="../data/sorted_by_end.tsv",
                        help='按 end_time 排序的 TSV 文件路径')
    parser.add_argument('--start_time', default='2019-06-14 17:12:20', help='开始时间（格式 YYYY-MM-DD HH:MM:SS）')
    parser.add_argument('--end_time', default='2019-07-05 00:00:00', help='结束时间（格式 YYYY-MM-DD HH:MM:SS）')
    parser.add_argument('--accelerate', type=float, default=2400, help='时间流速（每秒模拟时间增加秒数，默认 60）')
    parser.add_argument('--no-sleep', action='store_true', help='禁用延迟，快速测试')
    parser.add_argument('--num_threads', type=int, default=8, help='线程数量（默认 1）')
    args = parser.parse_args()

    simulate_logs(args.input, args.output, args.sorted_file, args.start_time, args.end_time, args.accelerate,
                  args.no_sleep, args.num_threads)