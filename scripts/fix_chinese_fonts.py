#!/usr/bin/env python3
"""
修复中文字体显示问题的脚本
重新生成所有图表以确保中文显示正确
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_fonts():
    """设置中文字体"""
    # 查找系统中可用的中文字体
    chinese_font_names = []
    for font in fm.fontManager.ttflist:
        font_name = font.name
        if any(keyword in font_name.lower() for keyword in ['droid', 'noto', 'wenquanyi', 'simhei', 'microsoft yahei', 'dejavu']):
            chinese_font_names.append(font_name)
    
    # 按优先级排序
    priority_fonts = ['Droid Sans Fallback', 'Noto Sans CJK', 'WenQuanYi', 'SimHei', 'Microsoft YaHei']
    final_fonts = []
    
    for pf in priority_fonts:
        for cf in chinese_font_names:
            if pf.lower() in cf.lower() and cf not in final_fonts:
                final_fonts.append(cf)
                break
    
    # 添加备用字体
    final_fonts.extend(['DejaVu Sans', 'Liberation Sans', 'sans-serif'])
    
    # 设置matplotlib字体
    plt.rcParams['font.sans-serif'] = final_fonts
    plt.rcParams['axes.unicode_minus'] = False
    
    print("已设置中文字体:")
    for i, font in enumerate(final_fonts[:5]):
        print(f"  {i+1}. {font}")
    
    return final_fonts

def regenerate_all_plots():
    """重新生成所有图表"""
    import subprocess
    import os
    
    # 设置工作目录
    script_dir = "/home/daniel/NGW/data/daughter/scripts"
    os.chdir(script_dir)
    
    scripts = [
        "02_nir_data_visualization.py",
        "03_nir_spectral_analysis.py", 
        "04_rice_quality_prediction.py"
    ]
    
    print("开始重新生成图表...")
    
    for script in scripts:
        print(f"\n正在运行 {script}...")
        try:
            result = subprocess.run(['python3', script], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ {script} 运行成功")
            else:
                print(f"❌ {script} 运行失败:")
                print(result.stderr)
        except subprocess.TimeoutExpired:
            print(f"⏰ {script} 运行超时")
        except Exception as e:
            print(f"❌ {script} 运行出错: {e}")

def main():
    """主函数"""
    print("=== 修复中文字体显示问题 ===")
    
    # 1. 设置字体
    fonts = setup_chinese_fonts()
    
    # 2. 测试字体显示
    print("\n测试中文字体显示...")
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3], [1, 4, 2], 'o-', linewidth=2, markersize=8)
    plt.xlabel('波长 (nm)')
    plt.ylabel('吸光度')
    plt.title('近红外光谱数据分析 - 字体测试')
    plt.grid(True, alpha=0.3)
    
    test_file = "/home/daniel/NGW/data/daughter/results/font_test.png"
    plt.savefig(test_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"字体测试图保存到: {test_file}")
    
    # 3. 询问是否重新生成所有图表
    print("\n是否重新生成所有图表以修复中文显示? (这将需要几分钟时间)")
    print("如果需要，请手动运行以下命令:")
    print("cd /home/daniel/NGW/data/daughter/scripts")
    print("python3 02_nir_data_visualization.py")
    print("python3 03_nir_spectral_analysis.py") 
    print("python3 04_rice_quality_prediction.py")
    
    print("\n=== 字体修复完成 ===")

if __name__ == "__main__":
    main()