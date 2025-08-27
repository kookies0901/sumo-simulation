#!/usr/bin/env python3
"""
测试图片拼接工具的新标签功能
1. 自定义标签位置（上端/下端）
2. 自定义起始编号
"""

import os
import subprocess
import sys
import glob

def create_test_images():
    """创建简单的测试图片"""
    from PIL import Image, ImageDraw
    
    test_dir = "analysis/charts/test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
    image_paths = []
    
    for i, color in enumerate(colors):
        img = Image.new('RGB', (400, 300), color)
        draw = ImageDraw.Draw(img)
        
        # 添加一些简单的文字标识
        draw.text((150, 130), f"Test {i+1}", fill='white', stroke_width=2, stroke_fill='black')
        
        path = os.path.join(test_dir, f"test_{i+1}.png")
        img.save(path)
        image_paths.append(path)
        print(f"✅ 创建测试图片: {path}")
    
    return image_paths

def test_label_features():
    """测试新的标签功能"""
    
    print("🧪 测试图片拼接工具的新标签功能")
    print("=" * 60)
    
    # 创建测试图片
    print("📷 创建测试图片...")
    test_images = create_test_images()
    
    if len(test_images) < 4:
        print("❌ 创建测试图片失败")
        return False
    
    # 测试用例
    test_cases = [
        {
            "name": "标签在顶部（默认）",
            "args": test_images[:4] + ["-o", "analysis/charts/test_labels_top.png", "-c", "2", "--labels"],
            "output": "analysis/charts/test_labels_top.png"
        },
        {
            "name": "标签在底部",
            "args": test_images[:4] + ["-o", "analysis/charts/test_labels_bottom.png", "-c", "2", "--labels", "--label-position", "bottom"],
            "output": "analysis/charts/test_labels_bottom.png"
        },
        {
            "name": "从c开始编号（顶部）",
            "args": test_images[:4] + ["-o", "analysis/charts/test_labels_start_c.png", "-c", "2", "--labels", "--label-start", "2"],
            "output": "analysis/charts/test_labels_start_c.png"
        },
        {
            "name": "从d开始编号（底部）",
            "args": test_images[:6] + ["-o", "analysis/charts/test_labels_start_d_bottom.pdf", "-c", "3", "--labels", "--label-position", "bottom", "--label-start", "3"],
            "output": "analysis/charts/test_labels_start_d_bottom.pdf"
        },
        {
            "name": "matplotlib版本（底部，从b开始）",
            "args": test_images[:4] + ["-o", "analysis/charts/test_matplotlib_bottom_b.pdf", "-c", "2", "--labels", "--matplotlib", "--label-position", "bottom", "--label-start", "1", "-t", "Matplotlib Test"],
            "output": "analysis/charts/test_matplotlib_bottom_b.pdf"
        },
        {
            "name": "复杂测试：6图，三列，从f开始，顶部",
            "args": test_images + ["-o", "analysis/charts/test_complex.png", "-c", "3", "--labels", "--label-start", "5", "-t", "Complex Layout Test"],
            "output": "analysis/charts/test_complex.png"
        }
    ]
    
    script_path = "analysis/scripts/image_combiner.py"
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试 {i}/{len(test_cases)}: {test_case['name']}")
        
        # 构建命令
        cmd = ["python", script_path] + test_case["args"]
        print(f"🔧 命令: {' '.join(cmd)}")
        
        try:
            # 运行命令
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # 检查输出文件是否存在
                if os.path.exists(test_case["output"]):
                    file_size = os.path.getsize(test_case["output"])
                    print(f"✅ 成功! 输出文件: {test_case['output']} ({file_size} bytes)")
                    success_count += 1
                else:
                    print(f"❌ 失败: 输出文件不存在 {test_case['output']}")
            else:
                print(f"❌ 命令执行失败:")
                print(f"   错误输出: {result.stderr}")
                if result.stdout:
                    print(f"   标准输出: {result.stdout}")
                
        except subprocess.TimeoutExpired:
            print(f"❌ 命令超时")
        except Exception as e:
            print(f"❌ 执行异常: {e}")
    
    print(f"\n📊 测试结果: {success_count}/{len(test_cases)} 成功")
    
    if success_count == len(test_cases):
        print("🎉 所有新功能测试通过!")
        print("\n📁 生成的测试文件:")
        for test_case in test_cases:
            if os.path.exists(test_case["output"]):
                print(f"   📄 {test_case['output']}")
        
        print("\n💡 功能测试说明:")
        print("   🔸 标签位置: --label-position top|bottom")
        print("   🔸 起始编号: --label-start N (0=a, 1=b, 2=c, ...)")
        print("   🔸 组合使用: 可同时设置位置和起始编号")
        
        return True
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
        return False

def cleanup_test_files():
    """清理测试文件"""
    import shutil
    
    test_dir = "analysis/charts/test_images"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"🧹 清理测试图片目录: {test_dir}")
    
    # 清理生成的测试文件
    test_files = glob.glob("analysis/charts/test_*.png") + glob.glob("analysis/charts/test_*.pdf")
    for file in test_files:
        try:
            os.remove(file)
            print(f"🧹 删除测试文件: {file}")
        except:
            pass

def show_feature_help():
    """显示新功能的详细说明"""
    print("🆕 图片拼接工具新功能说明")
    print("=" * 50)
    print()
    print("1️⃣ 自定义标签位置:")
    print("   --label-position top     # 标签在图片上方（默认）")
    print("   --label-position bottom  # 标签在图片下方")
    print()
    print("2️⃣ 自定义起始编号:")
    print("   --label-start 0   # 从 (a) 开始（默认）")
    print("   --label-start 1   # 从 (b) 开始")
    print("   --label-start 2   # 从 (c) 开始")
    print("   --label-start 3   # 从 (d) 开始")
    print("   ...")
    print()
    print("3️⃣ 组合使用示例:")
    print("   # 底部显示，从c开始编号")
    print("   python image_combiner.py *.png -o result.pdf -c 2 \\")
    print("     --labels --label-position bottom --label-start 2")
    print()
    print("   # 顶部显示，从e开始编号")
    print("   python image_combiner.py images/ -o output.png -c 3 \\")
    print("     --labels --label-position top --label-start 4")
    print()
    print("4️⃣ 适用场景:")
    print("   📖 多子图论文插图制作")
    print("   📊 按特定编号顺序排列图表")
    print("   📋 分组显示相关图片")
    print("   🎨 自定义标签位置以适应图片内容")

def main():
    """主函数"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            show_feature_help()
            return 0
        elif sys.argv[1] == "--cleanup":
            cleanup_test_files()
            return 0
    
    # 确保在正确的目录中
    if not os.path.exists("analysis/scripts/image_combiner.py"):
        print("❌ 请在项目根目录运行此脚本")
        return 1
    
    # 创建输出目录
    os.makedirs("analysis/charts", exist_ok=True)
    
    # 运行测试
    success = test_label_features()
    
    if success:
        print(f"\n💡 下一步:")
        print(f"   python analysis/scripts/test_label_features.py --help     # 查看功能说明")
        print(f"   python analysis/scripts/test_label_features.py --cleanup  # 清理测试文件")
        print(f"   python analysis/scripts/image_combiner.py --help         # 查看完整帮助")
        return 0
    else:
        return 1

if __name__ == '__main__':
    exit(main())
