import sys
import os

def compare_files(file1_path, file2_path):
    different_lines = []

    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

            min_lines = min(len(lines1), len(lines2))
            print(min_lines)

            for i in range(min_lines):
                if lines1[i] != lines2[i]:
                    different_lines.append(i + 1)

            if len(lines1) != len(lines2):
                print(f"文件行数不一致: {file1_path} 有 {len(lines1)} 行, {file2_path} 有 {len(lines2)} 行")

                if len(lines1) > len(lines2):
                    for i in range(min_lines, len(lines1)):
                        different_lines.append(i + 1)
                else:
                    for i in range(min_lines, len(lines2)):
                        different_lines.append(i + 1)
    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

    return different_lines


def main():
    if len(sys.argv) != 3:
        print("用法: python py_diff.py <文件1> <文件2>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    
    if not os.path.exists(file1_path):
        print(f"错误： 文件不存在 - {file1_path}")
        sys.exit(1)

    if not os.path.exists(file2_path):
        print(f"错误: 文件不存在 - {file2_path}")
        sys.exit(1)

    if not file2_path.endswith('.py') or not file1_path.endswith('.py'):
        print("警告： 至少有一个文件不是.py后缀的Python文件")

    different_lines = compare_files(file1_path, file2_path)

    if different_lines:
        print(f"发现 {len(different_lines)} 行内容不一致")
        for line_num in different_lines:
            print(f"行号 {line_num}")
    else:
        print("两个文件的内容完全一致")

if __name__ == "__main__":
    main()