import os

# --- 配置 ---
# 要扫描的根目录，'.' 表示当前目录
ROOT_DIR = '.'
# 输出文件名
OUTPUT_FILE = 'result.txt'
# 要忽略的目录（使用集合以提高查找效率）
IGNORE_DIRS = {'.git', '__pycache__', 'venv', '.vscode', 'node_modules','migrations'}
# 要忽略的文件
IGNORE_FILES = {'.DS_Store', OUTPUT_FILE,'遍历文件.py'} # 确保不把输出文件本身包含进去

def generate_file_tree(root_dir, ignore_dirs, ignore_files):
    """生成项目文件树结构"""
    tree_lines = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # 在遍历前，从dirs列表中移除要忽略的目录
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        # 计算当前深度，用于生成前缀
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        
        # 添加目录名到树
        # os.path.basename(root) 用于获取当前目录名
        tree_lines.append(f"{indent}📂 {os.path.basename(root)}/")

        # 添加文件到树
        sub_indent = ' ' * 4 * (level + 1)
        for f in sorted(files): # 对文件进行排序
            if f not in ignore_files:
                tree_lines.append(f"{sub_indent}📄 {f}")
                
    return "\n".join(tree_lines)

def get_python_file_contents(root_dir, ignore_dirs):
    """获取所有.py文件的内容并格式化"""
    py_contents = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # 同样，忽略指定目录
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in sorted(files):
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                # 使用相对路径，让输出更清晰
                relative_path = os.path.relpath(file_path, root_dir)
                
                header = f"####{relative_path}####"
                footer = "####文件结束####"
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    py_contents.append(f"{header}\n{content}\n{footer}\n")
                except Exception as e:
                    py_contents.append(f"{header}\n无法读取文件内容: {e}\n{footer}\n")
                    
    return "\n".join(py_contents)

def main():
    """主函数，执行所有操作"""
    print("开始生成项目文件树...")
    file_tree = generate_file_tree(ROOT_DIR, IGNORE_DIRS, IGNORE_FILES)
    
    print("开始读取所有.py文件内容...")
    python_contents = get_python_file_contents(ROOT_DIR, IGNORE_DIRS)
    
    print(f"正在将结果写入 {OUTPUT_FILE}...")
    
    # 将所有内容合并写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("======= 项目文件树 =======\n\n")
        f.write(file_tree)
        f.write("\n\n========================\n\n")
        f.write("======= Python文件内容 =======\n\n")
        f.write(python_contents)
        
    print(f"✅ 成功！项目结构和代码已保存到 {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
