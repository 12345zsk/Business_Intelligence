import pandas as pd
import argparse
import os


def detect_duplicate_end_time(input_file):
    try:
        df = pd.read_csv(input_file, sep='\t', encoding='utf-8')
        if 'end' not in df.columns:
            raise ValueError("TSV 文件必须包含 'end' 列")

        end_counts = df['end'].value_counts()
        duplicates = end_counts[end_counts > 1]

        if duplicates.empty:
            return 0
        return duplicates.sum() - len(duplicates)

    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 未找到")
        return 0
    except pd.errors.EmptyDataError:
        print(f"错误：文件 '{input_file}' 为空或格式不正确")
        return 0
    except ValueError as e:
        print(f"错误：{str(e)}")
        return 0
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='检测 TSV 文件中 end 列的重复行数。')
    parser.add_argument('--input', default="sorted_by_end.tsv", help='输入 TSV 文件路径（默认 simulate_log.tsv）')
    args = parser.parse_args()

    duplicate_count = detect_duplicate_end_time(args.input)
    print(f"重复的 end 时间行数：{duplicate_count}")