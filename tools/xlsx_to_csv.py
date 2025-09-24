#!/usr/bin/env python3
"""
Excel文件转CSV文件脚本
支持转换单个xlsx文件或目录下的所有xlsx文件
"""

import sys
import pandas as pd
from pathlib import Path

def convert_single_file(xlsx_path, csv_path):
    """
    转换单个xlsx文件为csv文件
    
    Args:
        xlsx_path (Path): xlsx文件路径
        csv_path (Path): csv文件输出路径
    """
    try:
        print(f"正在转换: {xlsx_path.name}")
        
        # 读取xlsx文件
        df = pd.read_excel(xlsx_path)
        
        # 确保输出目录存在
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为csv文件（使用UTF-8编码）
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"转换成功: {csv_path}")
        return True
        
    except Exception as e:
        print(f"转换失败 {xlsx_path.name}: {str(e)}")
        return False

def convert_directory(data_dir, output_dir=None):
    """
    将指定目录下的所有xlsx文件转换为csv文件
    
    Args:
        data_dir (Path): 包含xlsx文件的目录路径
        output_dir (Path): 输出csv文件的目录路径，如果为None则输出到原目录
    """
    if not data_dir.exists():
        print(f"错误：数据目录 {data_dir} 不存在")
        return False
    
    # 如果没有指定输出目录，使用输入目录
    if output_dir is None:
        output_path = data_dir
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有xlsx文件
    xlsx_files = list(data_dir.glob("*.xlsx"))
    
    if not xlsx_files:
        print(f"在目录 {data_dir} 中未找到xlsx文件")
        return False
    
    success_count = 0
    for xlsx_file in xlsx_files:
        csv_filename = xlsx_file.stem + ".csv"
        csv_path = output_path / csv_filename
        
        if convert_single_file(xlsx_file, csv_path):
            success_count += 1
    
    print(f"\n转换完成！成功转换 {success_count}/{len(xlsx_files)} 个文件")
    return success_count > 0

def main():
    """主函数"""
    if len(sys.argv) == 1:
        # 默认模式：转换默认目录下的所有xlsx文件
        data_dir = Path("/home/daniel/NGW/data/daughter/data")
        print(f"数据目录: {data_dir}")
        convert_directory(data_dir)
        
    elif len(sys.argv) == 2:
        input_path = Path(sys.argv[1])
        
        if input_path.suffix.lower() == '.xlsx':
            # 单文件模式：转换单个xlsx文件
            csv_path = input_path.with_suffix('.csv')
            print(f"输入文件: {input_path}")
            print(f"输出文件: {csv_path}")
            convert_single_file(input_path, csv_path)
        else:
            # 目录模式：转换目录下所有xlsx文件
            print(f"数据目录: {input_path}")
            convert_directory(input_path)
            
    elif len(sys.argv) == 3:
        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
        
        if input_path.suffix.lower() == '.xlsx':
            # 单文件模式：指定输出文件
            print(f"输入文件: {input_path}")
            print(f"输出文件: {output_path}")
            convert_single_file(input_path, output_path)
        else:
            # 目录模式：指定输出目录
            print(f"数据目录: {input_path}")
            print(f"输出目录: {output_path}")
            convert_directory(input_path, output_path)
    
    else:
        print("用法:")
        print("  python xlsx_to_csv.py                    # 转换默认目录下所有xlsx文件")
        print("  python xlsx_to_csv.py <目录>              # 转换指定目录下所有xlsx文件")
        print("  python xlsx_to_csv.py <xlsx文件>          # 转换单个xlsx文件")
        print("  python xlsx_to_csv.py <输入> <输出>       # 指定输入和输出路径")

if __name__ == "__main__":
    main()