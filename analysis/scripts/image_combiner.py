#!/usr/bin/env python3
"""
专业图片拼接工具 (Professional Image Combiner) - 完全修正版
支持将多个PNG图片按指定布局拼接为PDF或PNG文件

修正内容：
- 分离水平和垂直间距控制
- 智能尺寸调整：保持比例，避免白色填充
- 灵活的间距控制，适配标签显示
"""

import os
import argparse
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

class ImageCombiner:
    def __init__(self):
        """初始化图片拼接器"""
        self.supported_formats = ['.png', '.jpg', '.jpeg']
        self.default_h_spacing = 20  # 水平间距
        self.default_v_spacing = 20  # 垂直间距
        self.default_margin = 40   # 边距
        
    def collect_images(self, inputs):
        """收集输入的图片文件"""
        image_files = []
        
        for input_path in inputs:
            if os.path.isdir(input_path):
                # 文件夹输入 - 收集所有PNG文件
                pattern = os.path.join(input_path, "*.png")
                folder_images = glob.glob(pattern)
                folder_images.sort()  # 按文件名排序
                image_files.extend(folder_images)
                print(f"📁 从文件夹 {input_path} 收集到 {len(folder_images)} 张图片")
                
            elif os.path.isfile(input_path):
                # 单个文件输入
                if any(input_path.lower().endswith(ext) for ext in self.supported_formats):
                    image_files.append(input_path)
                    print(f"🖼️ 添加图片: {os.path.basename(input_path)}")
                else:
                    print(f"⚠️ 跳过不支持的文件格式: {input_path}")
            else:
                print(f"❌ 路径不存在: {input_path}")
        
        print(f"📊 总共收集到 {len(image_files)} 张图片")
        return image_files
    
    def load_and_validate_images(self, image_files):
        """加载并验证图片"""
        valid_images = []
        image_info = []
        
        for i, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                valid_images.append(img)
                image_info.append({
                    'path': img_path,
                    'name': os.path.basename(img_path),
                    'size': img.size,
                    'index': i
                })
                
                print(f"✅ 加载图片 {i+1}: {os.path.basename(img_path)} ({img.size[0]}×{img.size[1]})")
                
            except Exception as e:
                print(f"❌ 加载图片失败 {img_path}: {e}")
                continue
        
        return valid_images, image_info
    
    def calculate_layout_grid(self, num_images, columns):
        """计算网格布局"""
        if columns is None:
            # 自动确定列数
            if num_images <= 2:
                columns = num_images
            elif num_images <= 6:
                columns = 2
            elif num_images <= 12:
                columns = 3
            else:
                columns = 4
        
        rows = (num_images + columns - 1) // columns
        return rows, columns
    
    def auto_crop_white_borders(self, image, threshold=250):
        """自动裁剪图片的白色边框"""
        import numpy as np
        
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            # RGB或RGBA图片，只检查RGB通道
            rgb_array = img_array[:, :, :3]
        else:
            rgb_array = img_array
            
        height, width = rgb_array.shape[:2]
        
        # 找内容边界（非白色区域）
        # 检查左边界
        for left in range(width):
            if np.any(rgb_array[:, left] < threshold):
                break
        else:
            left = 0
            
        # 检查右边界
        for right in range(width-1, -1, -1):
            if np.any(rgb_array[:, right] < threshold):
                break
        else:
            right = width-1
            
        # 检查上边界
        for top in range(height):
            if np.any(rgb_array[top, :] < threshold):
                break
        else:
            top = 0
            
        # 检查下边界  
        for bottom in range(height-1, -1, -1):
            if np.any(rgb_array[bottom, :] < threshold):
                break
        else:
            bottom = height-1
            
        # 裁剪图片
        if left < right and top < bottom:
            cropped = image.crop((left, top, right+1, bottom+1))
            return cropped
        else:
            # 如果没有找到有效内容，返回原图
            return image
    
    def resize_images_proportional(self, images, target_height=None, uniform_width=False):
        """按比例调整图片尺寸，避免白色填充"""
        if not images:
            return []
        
        if target_height is None:
            # 使用所有图片的平均高度
            target_height = sum(img.size[1] for img in images) // len(images)
        
        resized_images = []
        print(f"🔧 按比例调整图片尺寸，目标高度: {target_height}px")
        
        for i, img in enumerate(images):
            # 计算新的宽度，保持宽高比
            aspect_ratio = img.size[0] / img.size[1]
            new_width = int(target_height * aspect_ratio)
            new_height = target_height
            
            # 调整图片大小
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append(resized_img)
            
            print(f"   图片{i+1}: {img.size} → {resized_img.size}")
        
        return resized_images
    
    def create_combined_image(self, images, image_info, columns=None, 
                            h_spacing=None, v_spacing=None, margin=None, title=None, add_labels=False,
                            label_position='top', label_start_index=0, keep_original_size=False, target_height=None):
        """创建拼接图片（支持分离的水平垂直间距）"""
        if not images:
            raise ValueError("没有有效的图片可供拼接")
        
        h_spacing = h_spacing if h_spacing is not None else self.default_h_spacing
        v_spacing = v_spacing if v_spacing is not None else self.default_v_spacing
        margin = margin or self.default_margin
        
        # 计算布局
        rows, columns = self.calculate_layout_grid(len(images), columns)
        print(f"📐 布局: {rows} 行 × {columns} 列")
        print(f"📏 间距设置: 水平 {h_spacing}px, 垂直 {v_spacing}px")
        
        # 处理图片尺寸
        if keep_original_size:
            print("🔧 保持原始图片尺寸")
            processed_images = images
        else:
            # 按比例调整到统一高度，避免白色填充
            processed_images = self.resize_images_proportional(images, target_height)
        
        # 按行列组织图片
        grid_images = []
        for row in range(rows):
            row_images = []
            for col in range(columns):
                idx = row * columns + col
                if idx < len(processed_images):
                    row_images.append(processed_images[idx])
                else:
                    row_images.append(None)  # 占位符
            grid_images.append(row_images)
        
        # 计算每行的最大高度和每列的最大宽度
        row_heights = []
        col_widths = [0] * columns
        
        for row_idx, row_images in enumerate(grid_images):
            max_height = 0
            for col_idx, img in enumerate(row_images):
                if img is not None:
                    col_widths[col_idx] = max(col_widths[col_idx], img.size[0])
                    max_height = max(max_height, img.size[1])
            row_heights.append(max_height)
        
        # 计算画布尺寸
        canvas_width = sum(col_widths) + (columns - 1) * h_spacing + 2 * margin
        title_height = 80 if title else 0
        
        # 标签空间（根据字体大小动态调整）
        if add_labels:
            label_font_size = getattr(self, 'label_font_size', 24)
            label_space = max(50, label_font_size + 30)  # 至少50px，大字体需要更多空间
            if label_position == 'bottom':
                top_label_height = 0
                bottom_label_height = label_space
            else:  # top
                top_label_height = label_space
                bottom_label_height = 0
        else:
            top_label_height = 0
            bottom_label_height = 0
        
        canvas_height = sum(row_heights) + (rows - 1) * v_spacing + 2 * margin + title_height + top_label_height + bottom_label_height
        
        print(f"📐 画布尺寸: {canvas_width} × {canvas_height}")
        
        # 创建画布
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        draw = ImageDraw.Draw(canvas)
        
        # 添加标题
        if title:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
            
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (canvas_width - title_width) // 2
            draw.text((title_x, margin // 2), title, fill='black', font=font)
        
        # 准备标签字体
        if add_labels:
            label_font_size = getattr(self, 'label_font_size', 24)  # 默认24px，比原来大很多
            try:
                label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", label_font_size)
            except:
                try:
                    label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", label_font_size)
                except:
                    try:
                        label_font = ImageFont.truetype("arial.ttf", label_font_size)
                    except:
                        try:
                            # 尝试加载稍大的默认字体
                            label_font = ImageFont.load_default(size=label_font_size)
                        except:
                            label_font = ImageFont.load_default()
        
        # 放置图片
        current_y = margin + title_height + top_label_height
        img_index = 0
        
        for row_idx, row_images in enumerate(grid_images):
            current_x = margin
            
            for col_idx, img in enumerate(row_images):
                if img is not None:
                    # 在列内居中对齐
                    x_offset = (col_widths[col_idx] - img.size[0]) // 2
                    y_offset = (row_heights[row_idx] - img.size[1]) // 2
                    
                    paste_x = current_x + x_offset
                    paste_y = current_y + y_offset
                    
                    canvas.paste(img, (paste_x, paste_y))
                    
                    # 添加字母编号
                    if add_labels:
                        letter = chr(ord('a') + img_index + label_start_index)
                        label_text = f"({letter})"
                        
                        label_bbox = draw.textbbox((0, 0), label_text, font=label_font)
                        label_width = label_bbox[2] - label_bbox[0]
                        label_height = label_bbox[3] - label_bbox[1]
                        label_x = paste_x + (img.size[0] - label_width) // 2
                        
                        if label_position == 'bottom':
                            label_y = paste_y + img.size[1] + 12
                        else:  # top
                            label_y = paste_y - label_height - 15
                        
                        # 添加白色背景，确保标签可见（根据实际字体大小调整）
                        padding = 8
                        label_bg_bbox = [
                            label_x - padding, 
                            label_y - padding//2, 
                            label_x + label_width + padding, 
                            label_y + label_height + padding
                        ]
                        draw.rectangle(label_bg_bbox, fill='white', outline='lightgray', width=1)
                        draw.text((label_x, label_y), label_text, fill='black', font=label_font)
                    
                    img_index += 1
                
                current_x += col_widths[col_idx] + h_spacing
            
            current_y += row_heights[row_idx] + v_spacing
        
        return canvas
    
    def create_matplotlib_figure(self, images, image_info, columns=None, title=None, add_labels=False,
                                label_position='top', label_start_index=0):
        """使用matplotlib创建专业图表"""
        if not images:
            raise ValueError("没有有效的图片可供拼接")
        
        # 计算布局
        rows, columns = self.calculate_layout_grid(len(images), columns)
        
        # 创建figure - 为标签预留更多空间
        fig_width = columns * 4 + 1
        extra_height = 1.5 if add_labels else 1 if title else 0.5
        fig_height = rows * 3.5 + extra_height
        
        fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height))
        
        # 处理单个子图的情况
        if rows == 1 and columns == 1:
            axes = [axes]
        elif rows == 1 or columns == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # 设置样式
        plt.style.use('default')
        fig.patch.set_facecolor('white')
        
        # 添加主标题
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # 显示图片
        for i, img in enumerate(images):
            if i < len(axes):
                axes[i].imshow(np.array(img))
                axes[i].axis('off')
                
                # 添加字母编号（如果启用）
                if add_labels:
                    letter = chr(ord('a') + i + label_start_index)
                    label_text = f"({letter})"
                    
                    if label_position == 'bottom':
                        x_pos, y_pos = 0.5, 0.05
                        va = 'bottom'
                    else:
                        x_pos, y_pos = 0.5, 0.95
                        va = 'top'
                    
                    axes[i].text(x_pos, y_pos, label_text, transform=axes[i].transAxes,
                               fontsize=14, fontweight='normal', va=va, ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # 隐藏多余的子图
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        # 调整子图间距
        if add_labels:
            plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=1.5)
        else:
            plt.tight_layout(pad=1.5)
        
        if title:
            if add_labels:
                plt.subplots_adjust(top=0.90, hspace=0.4)
            else:
                plt.subplots_adjust(top=0.92)
        
        return fig
    
    def save_output(self, canvas_or_fig, output_path, output_format='png', dpi=300):
        """保存输出文件"""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if output_format.lower() == 'pdf':
            if hasattr(canvas_or_fig, 'savefig'):
                # matplotlib figure
                canvas_or_fig.savefig(output_path, format='pdf', dpi=dpi, 
                                    bbox_inches='tight', facecolor='white')
            else:
                # PIL Image
                canvas_or_fig.save(output_path, 'PDF', resolution=dpi)
        else:
            if hasattr(canvas_or_fig, 'savefig'):
                # matplotlib figure
                canvas_or_fig.savefig(output_path, format='png', dpi=dpi, 
                                    bbox_inches='tight', facecolor='white')
            else:
                # PIL Image
                canvas_or_fig.save(output_path, 'PNG')
        
        print(f"✅ 输出文件已保存: {output_path}")
    
    def combine_images(self, inputs, output_path, columns=None, title=None, 
                      h_spacing=None, v_spacing=None, margin=None, output_format=None, 
                      use_matplotlib=False, dpi=300, add_labels=False,
                      label_position='top', label_start_index=0, auto_crop=False, 
                      keep_original_size=False, target_height=None, label_font_size=24):
        """主要的图片拼接函数"""
        
        print("🚀 开始图片拼接处理...")
        print(f"📁 输入: {inputs}")
        print(f"💾 输出: {output_path}")
        
        # 设置标签字体大小
        self.label_font_size = label_font_size
        if add_labels:
            print(f"🏷️ 标签字体大小: {label_font_size}px")
        
        if keep_original_size:
            print("🔧 启用原始尺寸模式")
        if auto_crop:
            print("✂️ 启用自动裁剪")
        
        # 确定输出格式
        if output_format is None:
            output_format = 'pdf' if output_path.lower().endswith('.pdf') else 'png'
        
        # 收集图片文件
        image_files = self.collect_images(inputs)
        if not image_files:
            print("❌ 没有找到有效的图片文件")
            return False
        
        # 加载图片
        images, image_info = self.load_and_validate_images(image_files)
        if not images:
            print("❌ 没有成功加载的图片")
            return False
        
        # 如果启用自动裁剪，裁剪掉白色边框
        if auto_crop:
            print("✂️ 自动裁剪白色边框...")
            cropped_images = []
            for i, img in enumerate(images):
                cropped = self.auto_crop_white_borders(img)
                cropped_images.append(cropped)
                original_size = img.size
                cropped_size = cropped.size
                print(f"   图片{i+1}: {original_size} → {cropped_size}")
            images = cropped_images
        
        try:
            if use_matplotlib:
                # 使用matplotlib创建
                result = self.create_matplotlib_figure(images, image_info, columns, title, 
                                                     add_labels, label_position, label_start_index)
            else:
                # 使用PIL创建
                result = self.create_combined_image(images, image_info, columns, 
                                                  h_spacing, v_spacing, margin, title, add_labels,
                                                  label_position, label_start_index, 
                                                  keep_original_size, target_height)
            
            # 保存结果
            self.save_output(result, output_path, output_format, dpi)
            
            # 显示统计信息
            print(f"\n📊 拼接完成统计:")
            print(f"   • 处理图片数量: {len(images)}")
            print(f"   • 布局设置: {columns}列" if columns else f"   • 自动布局")
            print(f"   • 水平间距: {h_spacing if h_spacing is not None else self.default_h_spacing}px")
            print(f"   • 垂直间距: {v_spacing if v_spacing is not None else self.default_v_spacing}px")
            print(f"   • 原始尺寸: {'启用' if keep_original_size else '禁用'}")
            print(f"   • 自动裁剪: {'启用' if auto_crop else '禁用'}")
            print(f"   • 字母编号: {'启用' if add_labels else '禁用'}")
            print(f"   • 输出格式: {output_format.upper()}")
            print(f"   • 输出路径: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 拼接过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='专业图片拼接工具 - 支持分离的水平垂直间距控制（完全修正版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分别设置水平和垂直间距
  python image_combiner.py *.png -o spaced.png --h-spacing 10 --v-spacing 30
  
  # 完全无间距拼接（保持比例）
  python image_combiner.py *.png -o tight.png --h-spacing 0 --v-spacing 0
  
  # 保持原始尺寸，自定义间距
  python image_combiner.py *.png -o original.png --h-spacing 5 --v-spacing 15 --keep-original-size
  
  # 带标签的拼接，增大垂直间距避免遮挡，设置大字体标签
  python image_combiner.py *.png -o labeled.png --labels --h-spacing 10 --v-spacing 60 --label-font-size 32
  
  # 小标签，紧凑布局
  python image_combiner.py *.png -o compact.png --labels --h-spacing 5 --v-spacing 35 --label-font-size 18
  
  # 自动裁剪后拼接
  python image_combiner.py *.png -o cropped.png --auto-crop --h-spacing 0 --v-spacing 5
  
  # 设置统一高度
  python image_combiner.py *.png -o uniform_height.png --target-height 400 --h-spacing 15 --v-spacing 25
        """
    )
    
    parser.add_argument('inputs', nargs='+', 
                       help='输入图片或文件夹路径（支持多个）')
    parser.add_argument('-o', '--output', required=True,
                       help='输出文件路径（.png或.pdf）')
    parser.add_argument('-c', '--columns', type=int, choices=[1, 2, 3, 4, 5, 6],
                       help='列数布局（1-6列，默认自动判断）')
    parser.add_argument('-t', '--title', 
                       help='图表标题')
    
    # 分离的间距控制
    parser.add_argument('--h-spacing', type=int, default=20,
                       help='水平间距（像素，默认20）')
    parser.add_argument('--v-spacing', type=int, default=20,
                       help='垂直间距（像素，默认20）')
    parser.add_argument('-s', '--spacing', type=int,
                       help='同时设置水平和垂直间距（会覆盖单独设置）')
    
    parser.add_argument('-m', '--margin', type=int, default=40,
                       help='边距（像素，默认40）')
    parser.add_argument('-f', '--format', choices=['png', 'pdf'],
                       help='输出格式（默认根据文件扩展名判断）')
    parser.add_argument('--matplotlib', action='store_true',
                       help='使用matplotlib生成学术级图表')
    parser.add_argument('--dpi', type=int, default=300,
                       help='输出分辨率（DPI，默认300）')
    
    # 标签控制
    parser.add_argument('--labels', action='store_true',
                       help='在每张图片上添加字母编号 (a), (b), (c)...')
    parser.add_argument('--label-position', choices=['top', 'bottom'], default='top',
                       help='标签位置：top（图片上方）或 bottom（图片下方），默认top')
    parser.add_argument('--label-start', type=int, default=0,
                       help='标签起始编号偏移（0=a, 1=b, 2=c...），默认0')
    parser.add_argument('--label-font-size', type=int, default=24,
                       help='标签字体大小（像素，默认24）')
    
    # 图片处理选项
    parser.add_argument('--auto-crop', action='store_true',
                       help='自动裁剪图片的白色边框')
    parser.add_argument('--keep-original-size', action='store_true',
                       help='保持图片原始尺寸，不做任何缩放')
    parser.add_argument('--target-height', type=int,
                       help='设置统一的目标高度（像素），保持宽高比')
    
    args = parser.parse_args()
    
    # 处理间距参数
    h_spacing = args.h_spacing
    v_spacing = args.v_spacing
    if args.spacing is not None:
        h_spacing = v_spacing = args.spacing
    
    # 创建拼接器
    combiner = ImageCombiner()
    
    # 执行拼接
    success = combiner.combine_images(
        inputs=args.inputs,
        output_path=args.output,
        columns=args.columns,
        title=args.title,
        h_spacing=h_spacing,
        v_spacing=v_spacing,
        margin=args.margin,
        output_format=args.format,
        use_matplotlib=args.matplotlib,
        dpi=args.dpi,
        add_labels=args.labels,
        label_position=args.label_position,
        label_start_index=args.label_start,
        auto_crop=args.auto_crop,
        keep_original_size=args.keep_original_size,
        target_height=args.target_height,
        label_font_size=args.label_font_size
    )
    
    if success:
        print(f"\n🎉 图片拼接成功完成!")
        return 0
    else:
        print(f"\n❌图片拼接失败!")
        return 1

if __name__ == '__main__':
    exit(main())