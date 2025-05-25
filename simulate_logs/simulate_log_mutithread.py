import pandas as pd
import time
import argparse
import os
import threading
import queue
from datetime import datetime, timedelta


def clean_field(value):
    if isinstance(value, str):
        return value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return value


def process_thread(thread_id, output_file, task_queue, stop_event):
    processed_rows = 0
    thread_output = f"{output_file}_thread_{thread_id}.tsv"
    try:
        with open(thread_output, 'w', newline='', encoding='utf-8') as f:
            columns = ['UserID', 'ClicknewsID', 'dwelltime', 'exposure_time', 'pos', 'neg', 'start', 'end',
                       'dwelltime_pos', 'thread_id']
            f.write('\t'.join(columns) + '\n')

            while not stop_event.is_set() or not task_queue.empty():
                try:
                    row = task_queue.get(timeout=0.5)
                    values = [str(row[col]) for col in row.index if col != 'end_timestamp'] + [str(thread_id)]
                    f.write('\t'.join(values) + '\n')
                    processed_rows += 1
                    task_queue.task_done()
                except queue.Empty:
                    continue

    except Exception as e:
        print(f"[{datetime.now()}] 线程 {thread_id} 错误：{str(e)}")

    print(f"[{datetime.now()}] 线程 {thread_id} 完成，共写入 {processed_rows} 行 -> {thread_output}")


def simulate_logs(input_file, output_file, sorted_file,
                  start_time=None, end_time=None, accelerate=1, no_sleep=False):
    start_real_time = time.time()
    num_threads = 4

    try:
        if os.path.exists(sorted_file):
            print(f"[{datetime.now()}] 读取排序文件：{sorted_file}")
            df = pd.read_csv(sorted_file, sep='\t', encoding='utf-8')
        else:
            print(f"[{datetime.now()}] 读取原始数据并排序")
            df = pd.read_csv(input_file, sep='\t', encoding='utf-8')
            for col in df.columns:
                df[col] = df[col].apply(clean_field)
            df['end_timestamp'] = pd.to_datetime(df['end'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            df.dropna(subset=['end_timestamp'], inplace=True)
            df = df.sort_values('end_timestamp')
            df.drop(columns=['end_timestamp']).to_csv(sorted_file, sep='\t', index=False, encoding='utf-8')

        if 'end_timestamp' not in df.columns:
            df['end_timestamp'] = pd.to_datetime(df['end'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            df.dropna(subset=['end_timestamp'], inplace=True)

        if start_time:
            start_time = pd.to_datetime(start_time)
            df = df[df['end_timestamp'] >= start_time]
        if end_time:
            end_time = pd.to_datetime(end_time)
            df = df[df['end_timestamp'] <= end_time]

        if df.empty:
            raise ValueError("时间范围内无可用数据")

        sim_start_time = start_time if start_time else df['end_timestamp'].iloc[0]
        sim_end_time = df['end_timestamp'].iloc[-1]

        print(f"[{datetime.now()}] 模拟从 {sim_start_time} 到 {sim_end_time} 开始...")

        task_queues = [queue.Queue() for _ in range(num_threads)]
        stop_event = threading.Event()
        threads = []

        for i in range(num_threads):
            t = threading.Thread(target=process_thread, args=(i, output_file, task_queues[i], stop_event))
            t.start()
            threads.append(t)

        sim_current_time = sim_start_time
        total_processed = 0

        while sim_current_time <= sim_end_time and not df.empty:
            current_batch = df[df['end_timestamp'] <= sim_current_time]
            df = df[df['end_timestamp'] > sim_current_time]

            if not current_batch.empty:
                rows = current_batch.to_dict(orient='records')
                if len(rows) == 1:
                    task_queues[0].put(current_batch.iloc[0])
                else:
                    for i, row_dict in enumerate(rows):
                        row = pd.Series(row_dict)
                        thread_id = i % num_threads
                        task_queues[thread_id].put(row)

                total_processed += len(rows)
                print(f"[{datetime.now()}] 模拟时间 {sim_current_time}，处理 {len(rows)} 行，总处理 {total_processed}")

            if not no_sleep:
                time.sleep(1)
            sim_current_time = sim_end_time if (sim_current_time + timedelta(seconds=accelerate)) > sim_end_time \
                else (sim_current_time + timedelta(seconds=accelerate))

        # 等待所有线程处理完任务
        for q in task_queues:
            q.join()
        stop_event.set()
        for t in threads:
            t.join()

        total_written = 0
        for i in range(num_threads):
            path = f"{output_file}_thread_{i}.tsv"
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f) - 1
                    print(f"[{datetime.now()}] {path} 共 {lines} 行")
                    total_written += lines

        print(f"[{datetime.now()}] 总写入行数：{total_written}")
        print(f"[{datetime.now()}] 所有线程完成写入。")

    except Exception as e:
        print(f"[{datetime.now()}] 错误：{str(e)}")
    finally:
        duration = time.time() - start_real_time
        print(f"[{datetime.now()}] 模拟总耗时：{duration:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模拟新闻点击日志 - 多线程可变处理')
    parser.add_argument('--input', default='../data/train.tsv')
    parser.add_argument('--output', default='../data/simulate_log_multi-thread')
    parser.add_argument('--sorted_file', default='../data/sorted_by_end.tsv')
    parser.add_argument('--start_time', default='2019-06-14 17:12:20')
    parser.add_argument('--end_time', default='2019-06-14 17:26:20')
    # parser.add_argument('--end_time', default='2019-07-05 00:00:00')
    parser.add_argument('--accelerate', type=float, default=2400)
    parser.add_argument('--no-sleep', action='store_true')
    args = parser.parse_args()

    simulate_logs(args.input, args.output, args.sorted_file, args.start_time,
                  args.end_time, args.accelerate, args.no_sleep)
