import pandas as pd
import time
import argparse
import os
import threading
import queue
from datetime import datetime, timedelta


def clean_field(value):
    """清理字段中的换行符和特殊字符"""
    if isinstance(value, str):
        return value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return value


def process_thread(thread_id, output_file, task_queue, stop_event):
    """线程处理函数，从独立队列获取行并写入独立文件"""
    processed_rows = 0
    thread_output = f"{output_file}_thread_{thread_id}.tsv"

    try:
        # 打开文件并写入表头
        with open(thread_output, 'w', newline='', encoding='utf-8') as f:
            columns = ['UserID', 'ClicknewsID', 'dwelltime', 'exposure_time', 'pos', 'neg', 'start', 'end',
                       'dwelltime_pos', 'thread_id']
            f.write('\t'.join(columns) + '\n')
            f.flush()

        # 循环处理任务队列，不仅在停止标记触发时退出，还要确保队列内任务全部写入
        while not stop_event.is_set() or not task_queue.empty():
            try:
                # 使用超时方式等待任务，避免一直阻塞
                row = task_queue.get(timeout=0.5)
                # 打开文件追加写入数据，注意这里无需加锁，因为各线程操作不同的文件
                with open(thread_output, 'a', newline='', encoding='utf-8') as f:
                    values = [str(row[col]) for col in row.index if col != 'end_timestamp'] + [str(thread_id)]
                    f.write('\t'.join(values) + '\n')
                task_queue.task_done()
                processed_rows += 1
                if processed_rows % 1000 == 0:
                    print(f"[{datetime.now()}] 线程 {thread_id} 已处理 {processed_rows} 行")
            except queue.Empty:
                continue
    except Exception as e:
        print(f"[{datetime.now()}] 线程 {thread_id} 发生错误：{str(e)}")

    print(f"[{datetime.now()}] 线程 {thread_id} 完成，总计处理 {processed_rows} 行，输出到 '{thread_output}'")


def simulate_logs(input_file="../data/train.tsv", output_file="../data/simulate_log_multi-thread",
                  sorted_file="../data/sorted_by_end.tsv",
                  start_time=None, end_time=None, accelerate=1, no_sleep=False):
    """
    模拟从 PENS 印象日志生成实时新闻点击日志流，同一时间多条记录分配到 4 个线程独立输出，不合并。

    参数：
        input_file (str): 输入 TSV 文件路径，默认为 '../data/train.tsv'
        output_file (str): 输出文件前缀，线程文件为 '{output_file}_thread_{0-3}.tsv'
        sorted_file (str): 按 end_time 排序的 TSV 文件路径，默认为 '../data/sorted_by_end.tsv'
        start_time (str): 模拟开始时间（格式 YYYY-MM-DD HH:MM:SS），默认为最早 end_time
        end_time (str): 模拟结束时间（格式 YYYY-MM-DD HH:MM:SS），默认为最晚 end_time
        accelerate (float): 时间流速（每秒模拟时间增加秒数），默认为 1
        no_sleep (bool): 是否禁用延迟（快速测试），默认为 False
    """
    start_real_time = time.time()
    num_threads = 4

    try:
        # 清空分配日志文件
        assignment_log_file = '../data/assignment_log.txt'
        if os.path.exists(assignment_log_file):
            os.remove(assignment_log_file)

        if os.path.exists(sorted_file):
            print(f"[{datetime.now()}] 检测到排序文件 '{sorted_file}'，直接读取...")
            df = pd.read_csv(sorted_file, sep='\t', encoding='utf-8')
            # 为后续使用还原 end_timestamp 列
            df['end_timestamp'] = pd.to_datetime(df['end'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        else:
            print(f"[{datetime.now()}] 排序文件 '{sorted_file}' 不存在，从 '{input_file}' 生成...")
            df = pd.read_csv(input_file, sep='\t', encoding='utf-8')

            required_columns = ['start', 'end']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"输入 TSV 文件必须包含以下列：{required_columns}")

            print(f"[{datetime.now()}] 清理字段中的换行符...")
            for col in df.columns:
                df[col] = df[col].apply(clean_field)

            print(f"[{datetime.now()}] 解析 end 时间戳并排序...")
            df['end_timestamp'] = pd.to_datetime(df['end'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            invalid_rows = df['end_timestamp'].isna()
            if invalid_rows.any():
                print(f"[{datetime.now()}] 无效时间戳行数：{invalid_rows.sum()}")
                df[invalid_rows][['end']].to_csv('../data/invalid_end_times.tsv', sep='\t', index=False)
            df = df[~invalid_rows]
            print(f"[{datetime.now()}] 有效数据行数：{len(df)}")

            df = df.sort_values('end_timestamp')
            print(f"[{datetime.now()}] 保存排序后的文件到 '{sorted_file}'...")
            df.drop(columns=['end_timestamp']).to_csv(sorted_file, sep='\t', index=False, encoding='utf-8',
                                                      lineterminator='\n')
            # 重新计算 end_timestamp 列供后续使用
            df['end_timestamp'] = pd.to_datetime(df['end'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

        print(f"[{datetime.now()}] 全部数据行数：{len(df)}")
        if start_time:
            start_time = pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S')
            df = df[df['end_timestamp'] >= start_time]
            print(f"[{datetime.now()}] 过滤 start_time 后行数：{len(df)}")
        if end_time:
            end_time = pd.to_datetime(end_time, format='%Y-%m-%d %H:%M:%S')
            df = df[df['end_timestamp'] <= end_time]
            print(f"[{datetime.now()}] 过滤 end_time 后行数：{len(df)}")

        if df.empty:
            raise ValueError("指定时间范围内的数据为空。")

        sim_start_time = start_time if start_time else df['end_timestamp'].iloc[0]
        sim_end_time = df['end_timestamp'].iloc[-1] if end_time is None else min(pd.to_datetime(end_time),
                                                                                 df['end_timestamp'].iloc[-1])
        print(f"[{datetime.now()}] 模拟时间跨度：{sim_start_time} 至 {sim_end_time}")
        print(f"[{datetime.now()}] 时间流速：{accelerate} 秒/秒，线程数：{num_threads}")

        # 为每个线程分配独立的任务队列
        task_queues = [queue.Queue() for _ in range(num_threads)]
        stop_event = threading.Event()
        threads = []

        # 启动各线程，注意这里每个线程只从自己的队列中取数据，不需要额外加锁
        for i in range(num_threads):
            thread = threading.Thread(
                target=process_thread,
                args=(i, output_file, task_queues[i], stop_event)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
            print(f"[{datetime.now()}] 启动线程 {i}")

        processed_rows = 0
        sim_current_time = sim_start_time
        assignment_buffer = []
        buffer_size = 1000  # 每 1000 行写入日志文件一次

        while sim_current_time <= sim_end_time and not df.empty:
            rows_to_write = df[df['end_timestamp'] <= sim_current_time]

            if not rows_to_write.empty:
                num = len(rows_to_write)
                processed_rows += num
                print(f"[{datetime.now()}] 当前时间 {sim_current_time} 处理 {num} 条记录，总处理 {processed_rows} 条")

                for idx, row in rows_to_write.iterrows():
                    thread_id = idx % num_threads
                    assignment_buffer.append(
                        f"[{datetime.now()}] 行索引 {idx} (UserID: {row['UserID']}) 分配到线程 {thread_id}\n")
                    if len(assignment_buffer) >= buffer_size:
                        with open(assignment_log_file, 'a', encoding='utf-8') as log_file:
                            log_file.writelines(assignment_buffer)
                        assignment_buffer = []
                    # 将任务放入对应线程的队列中
                    task_queues[thread_id].put(row)

                df = df[df['end_timestamp'] > sim_current_time]

            if not no_sleep:
                time.sleep(1)
            next_time = sim_current_time + timedelta(seconds=max(accelerate, 1))
            sim_current_time = min(next_time, sim_end_time)

        # 写入剩余的分配日志
        if assignment_buffer:
            with open(assignment_log_file, 'a', encoding='utf-8') as log_file:
                log_file.writelines(assignment_buffer)

        print(f"[{datetime.now()}] 剩余未处理行数：{len(df)}")

        # 等待所有任务队列中的任务处理完毕
        for q in task_queues:
            q.join()

        # 通知线程退出
        stop_event.set()
        for thread in threads:
            thread.join()

        total_rows = 0
        for i in range(num_threads):
            thread_file = f"{output_file}_thread_{i}.tsv"
            if os.path.exists(thread_file):
                with open(thread_file, 'r', encoding='utf-8') as f:
                    # 减去表头行
                    lines = sum(1 for _ in f) - 1
                    print(f"[{datetime.now()}] {thread_file} 行数：{lines}")
                    total_rows += lines
        print(f"[{datetime.now()}] 总输出行数：{total_rows}")

        print(f"[{datetime.now()}] 所有线程完成，生成以下输出文件：")
        for i in range(num_threads):
            print(f"  {output_file}_thread_{i}.tsv")
        print(f"[{datetime.now()}] 分配日志已写入 '{assignment_log_file}'")

        end_real_time = time.time()
        execution_time = end_real_time - start_real_time
        execution_time_formatted = str(timedelta(seconds=int(execution_time)))
        print(f"[{datetime.now()}] 程序执行时间：{execution_time:.2f} 秒（{execution_time_formatted}）")

    except FileNotFoundError:
        print(f"[{datetime.now()}] 错误：文件 '{input_file}' 未找到")
    except pd.errors.EmptyDataError:
        print(f"[{datetime.now()}] 错误：文件 '{input_file}' 为空或格式不正确")
    except ValueError as e:
        print(f"[{datetime.now()}] 错误：{str(e)}")
    except Exception as e:
        print(f"[{datetime.now()}] 发生错误：{str(e)}")
    finally:
        end_real_time = time.time()
        execution_time = end_real_time - start_real_time
        execution_time_formatted = str(timedelta(seconds=int(execution_time)))
        print(f"[{datetime.now()}] 程序执行时间：{execution_time:.2f} 秒（{execution_time_formatted}）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='从 PENS 印象日志模拟实时新闻点击日志流，同一时间多条记录分配到 4 个线程独立输出，不合并。')
    parser.add_argument('--input', default="../data/train.tsv", help='输入 TSV 文件路径')
    parser.add_argument('--output', default="../data/simulate_log_multi-thread", help='输出文件前缀')
    parser.add_argument('--sorted_file', default="../data/sorted_by_end.tsv", help='按 end_time 排序的 TSV 文件路径')
    parser.add_argument('--start_time', default='2019-06-14 17:12:20', help='开始时间（格式 YYYY-MM-DD HH:MM:SS）')
    # parser.add_argument('--end_time', default='2019-06-14 17:22:20', help='结束时间（格式 YYYY-MM-DD HH:MM:SS）')
    parser.add_argument('--end_time', default='2019-07-05 00:00:00', help='结束时间（格式 YYYY-MM-DD HH:MM:SS）')
    parser.add_argument('--accelerate', type=float, default=2400, help='时间流速（每秒模拟时间增加秒数，默认 2400）')
    parser.add_argument('--no-sleep', action='store_true', help='禁用延迟，快速测试')
    args = parser.parse_args()

    simulate_logs(args.input, args.output, args.sorted_file, args.start_time, args.end_time, args.accelerate,
                  args.no_sleep)
