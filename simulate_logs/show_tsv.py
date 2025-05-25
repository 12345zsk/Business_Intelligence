import sys
import pandas as pd
import argparse

def show_and_save_head(tsv_file='../data/train.tsv', n=5, output_tsv='../data/train_5.tsv'):
    """
    显示TSV文件的前n行并保存到TSV文件。

    参数:
        tsv_file (str): 输入TSV文件路径，默认为 '../../train.tsv'
        n (int): 要显示和保存的行数，默认为 5
        output_tsv (str): 输出TSV文件路径，默认为 'train_5.tsv'
    """
    try:
        # 读取TSV文件
        df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8')

        # 检查文件是否为空
        if df.empty:
            print(f"错误：文件 '{tsv_file}' 为空")
            return

        # 保存前n行到TSV文件
        df.head(n).to_csv(output_tsv, sep='\t', index=False, encoding='utf-8')
        print(f"\n已将前 {n} 行保存到 '{output_tsv}'")

    except FileNotFoundError:
        print(f"错误：文件 '{tsv_file}' 未找到")
    except pd.errors.EmptyDataError:
        print(f"错误：文件 '{tsv_file}' 为空或格式不正确")
    except Exception as e:
        print(f"发生错误：{str(e)}")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='显示TSV文件的前n行并保存到TSV文件')
    parser.add_argument('--tsv_file', nargs='?', default='../data/train.tsv', help='输入TSV文件路径')
    parser.add_argument('--n', type=int, nargs='?', default=5, help='要显示和保存的行数')
    parser.add_argument('--output_tsv', nargs='?', default='train_5.tsv', help='输出TSV文件路径')
    args = parser.parse_args()

    # 调用展示和保存函数
    show_and_save_head(args.tsv_file, args.n, args.output_tsv)

if __name__ == "__main__":
    main()