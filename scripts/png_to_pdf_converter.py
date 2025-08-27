#!/usr/bin/env python3
"""
PNG图片批量转PDF工具
将指定文件夹下的所有PNG图片合并到一个PDF文件中
支持自定义排序、页面布局和质量设置
"""

import os
import argparse
import glob
from PIL import Image
import re
from pathlib import Path

def natural_sort_key(text):
    """自然排序键函数，正确处理数字排序"""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', text)]

def convert_pngs_to_pdf(input_dir, output_pdf=None, sort_method='name', 
                       page_layout='single', quality=95, max_width=None, max_height=None):
    """
    将文件夹中的PNG图片转换为PDF
    
    参数:
    input_dir: 输入文件夹路径
    output_pdf: 输出PDF文件路径（可选）
    sort_method: 排序方式 ('name', 'natural', 'date', 'size')
    page_layout: 页面布局 ('single', 'double', 'grid')
    quality: 图片质量 (1-100)
    max_width/max_height: 最大尺寸限制
    """
    
    # 验证输入目录
    if not os.path.isdir(input_dir):
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    # 查找所有PNG文件
    png_pattern = os.path.join(input_dir, "*.png")
    png_files = glob.glob(png_pattern)
    
    if not png_files:
        print(f"❌ 在目录 {input_dir} 中未找到PNG文件")
        return None
    
    print(f"📊 找到 {len(png_files)} 个PNG文件")
    
    # 排序文件
    if sort_method == 'natural':
        png_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    elif sort_method == 'name':
        png_files.sort(key=lambda x: os.path.basename(x))
    elif sort_method == 'date':
        png_files.sort(key=lambda x: os.path.getmtime(x))
    elif sort_method == 'size':
        png_files.sort(key=lambda x: os.path.getsize(x))
    
    # 确定输出PDF路径
    if output_pdf is None:
        dir_name = os.path.basename(os.path.abspath(input_dir))
        output_pdf = os.path.join(input_dir, f"{dir_name}_combined.pdf")
    else:
        # 如果指定了输出路径但不是绝对路径，则相对于输入目录
        if not os.path.isabs(output_pdf):
            output_pdf = os.path.join(input_dir, output_pdf)
    
    print(f"📁 输出PDF: {output_pdf}")
    print(f"🔤 排序方式: {sort_method}")
    print(f"📄 页面布局: {page_layout}")
    
    try:
        # 处理图片
        processed_images = []
        
        for i, png_file in enumerate(png_files):
            print(f"🔄 处理 ({i+1}/{len(png_files)}): {os.path.basename(png_file)}")
            
            try:
                # 打开图片
                img = Image.open(png_file)
                
                # 转换为RGB格式（PDF需要）
                if img.mode in ('RGBA', 'LA', 'P'):
                    # 创建白色背景
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整图片尺寸
                if max_width or max_height:
                    img = resize_image(img, max_width, max_height)
                
                processed_images.append(img)
                
            except Exception as e:
                print(f"⚠️ 处理图片失败 {png_file}: {e}")
                continue
        
        if not processed_images:
            print("❌ 没有成功处理的图片")
            return None
        
        # 根据页面布局处理图片
        if page_layout == 'single':
            final_images = processed_images
        elif page_layout == 'double':
            final_images = create_double_layout(processed_images)
        elif page_layout == 'grid':
            final_images = create_grid_layout(processed_images)
        else:
            final_images = processed_images
        
        # 保存为PDF
        if final_images:
            # 第一张图片作为主图片
            first_img = final_images[0]
            other_imgs = final_images[1:] if len(final_images) > 1 else []
            
            # 保存PDF
            first_img.save(
                output_pdf, 
                "PDF", 
                resolution=100.0,
                save_all=True, 
                append_images=other_imgs,
                quality=quality,
                optimize=True
            )
            
            print(f"✅ PDF生成成功!")
            print(f"📄 总页数: {len(final_images)}")
            print(f"💾 文件大小: {get_file_size(output_pdf)}")
            print(f"📁 保存位置: {output_pdf}")
            
            return output_pdf
        
    except Exception as e:
        print(f"❌ PDF生成失败: {e}")
        return None

def resize_image(img, max_width=None, max_height=None):
    """调整图片尺寸，保持宽高比"""
    width, height = img.size
    
    if max_width and width > max_width:
        ratio = max_width / width
        width = max_width
        height = int(height * ratio)
    
    if max_height and height > max_height:
        ratio = max_height / height
        height = max_height
        width = int(width * ratio)
    
    if (width, height) != img.size:
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    
    return img

def create_double_layout(images):
    """创建双页布局（每页2张图）"""
    if len(images) <= 1:
        return images
    
    final_images = []
    
    for i in range(0, len(images), 2):
        if i + 1 < len(images):
            # 两张图片并排
            img1, img2 = images[i], images[i + 1]
            
            # 调整高度一致
            min_height = min(img1.size[1], img2.size[1])
            ratio1 = min_height / img1.size[1]
            ratio2 = min_height / img2.size[1]
            
            new_img1 = img1.resize((int(img1.size[0] * ratio1), min_height), Image.Resampling.LANCZOS)
            new_img2 = img2.resize((int(img2.size[0] * ratio2), min_height), Image.Resampling.LANCZOS)
            
            # 合并图片
            total_width = new_img1.size[0] + new_img2.size[0] + 20  # 20px间距
            combined = Image.new('RGB', (total_width, min_height), (255, 255, 255))
            combined.paste(new_img1, (0, 0))
            combined.paste(new_img2, (new_img1.size[0] + 20, 0))
            
            final_images.append(combined)
        else:
            # 单张图片
            final_images.append(images[i])
    
    return final_images

def create_grid_layout(images, grid_size=4):
    """创建网格布局（每页多张图）"""
    if len(images) <= 1:
        return images
    
    final_images = []
    
    for i in range(0, len(images), grid_size):
        batch = images[i:i + grid_size]
        
        if len(batch) == 1:
            final_images.append(batch[0])
            continue
        
        # 计算网格尺寸
        rows = 2 if len(batch) > 2 else 1
        cols = 2 if len(batch) > 1 else 1
        
        # 调整每张图片尺寸
        max_img_width = 800  # 每张图片最大宽度
        max_img_height = 600  # 每张图片最大高度
        
        resized_batch = []
        for img in batch:
            resized_img = resize_image(img, max_img_width, max_img_height)
            resized_batch.append(resized_img)
        
        # 创建网格
        grid_width = max_img_width * cols + 20 * (cols + 1)
        grid_height = max_img_height * rows + 20 * (rows + 1)
        
        grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        for idx, img in enumerate(resized_batch):
            row = idx // cols
            col = idx % cols
            x = 20 + col * (max_img_width + 20)
            y = 20 + row * (max_img_height + 20)
            
            # 居中放置图片
            img_x = x + (max_img_width - img.size[0]) // 2
            img_y = y + (max_img_height - img.size[1]) // 2
            
            grid_img.paste(img, (img_x, img_y))
        
        final_images.append(grid_img)
    
    return final_images

def get_file_size(file_path):
    """获取文件大小的友好显示"""
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def batch_convert_directories(base_dir, recursive=True):
    """批量处理多个目录"""
    print(f"🔍 批量处理目录: {base_dir}")
    
    if recursive:
        # 递归查找所有包含PNG的子目录
        png_dirs = set()
        for root, dirs, files in os.walk(base_dir):
            if any(f.lower().endswith('.png') for f in files):
                png_dirs.add(root)
    else:
        # 只处理直接子目录
        png_dirs = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                if any(f.lower().endswith('.png') for f in os.listdir(item_path)):
                    png_dirs.append(item_path)
    
    print(f"📂 找到 {len(png_dirs)} 个包含PNG的目录")
    
    results = []
    for i, png_dir in enumerate(sorted(png_dirs), 1):
        print(f"\n[{i}/{len(png_dirs)}] 处理目录: {os.path.basename(png_dir)}")
        try:
            pdf_path = convert_pngs_to_pdf(png_dir, sort_method='natural')
            if pdf_path:
                results.append(pdf_path)
        except Exception as e:
            print(f"❌ 处理失败: {e}")
    
    print(f"\n🎉 批量处理完成! 成功生成 {len(results)} 个PDF文件")
    return results

def main():
    parser = argparse.ArgumentParser(description='将PNG图片批量转换为PDF')
    parser.add_argument('input_dir', help='输入目录路径')
    parser.add_argument('-o', '--output', help='输出PDF文件路径（可选）')
    parser.add_argument('-s', '--sort', choices=['name', 'natural', 'date', 'size'], 
                       default='natural', help='排序方式 (默认: natural)')
    parser.add_argument('-l', '--layout', choices=['single', 'double', 'grid'], 
                       default='single', help='页面布局 (默认: single)')
    parser.add_argument('-q', '--quality', type=int, default=95, 
                       help='图片质量 1-100 (默认: 95)')
    parser.add_argument('--max-width', type=int, help='图片最大宽度')
    parser.add_argument('--max-height', type=int, help='图片最大高度')
    parser.add_argument('-b', '--batch', action='store_true', 
                       help='批量处理子目录')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='递归处理所有子目录')
    
    args = parser.parse_args()
    
    print("🔄 PNG转PDF工具启动")
    print(f"📁 输入目录: {args.input_dir}")
    
    try:
        if args.batch:
            # 批量处理模式
            batch_convert_directories(args.input_dir, args.recursive)
        else:
            # 单目录处理模式
            result = convert_pngs_to_pdf(
                input_dir=args.input_dir,
                output_pdf=args.output,
                sort_method=args.sort,
                page_layout=args.layout,
                quality=args.quality,
                max_width=args.max_width,
                max_height=args.max_height
            )
            
            if result:
                print(f"\n🎉 转换完成! PDF文件: {result}")
            else:
                print(f"\n❌ 转换失败!")
                return 1
    
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
