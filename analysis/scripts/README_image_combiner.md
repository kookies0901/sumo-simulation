# 专业图片拼接工具 (Professional Image Combiner)

## 功能概述

这是一个专业的图片拼接工具，专为学术论文和研究报告设计，支持将多个PNG图片按指定布局拼接为高质量的PDF或PNG文件。

## 主要特性

### ✨ 核心功能
- **灵活输入**: 支持文件夹批量输入和单个文件指定
- **多种布局**: 支持1-3列的自动或手动布局
- **智能编号**: 可选的 (a), (b), (c)... 学术图表编号，支持自定义位置和起始编号
- **双引擎**: PIL和matplotlib两种渲染引擎
- **高质量输出**: 支持PDF和PNG，300 DPI高分辨率

### 🎨 布局特性
- **自动缩放**: 智能调整图片尺寸保持一致性
- **专业间距**: 优化的行间距避免标签重叠
- **标题支持**: 可添加整体标题
- **无文件名**: 清洁的输出，不显示文件名

## 安装要求

```bash
pip install Pillow matplotlib numpy
```

## 使用方法

### 基本语法
```bash
python image_combiner.py [输入图片/文件夹] -o [输出文件] [选项]
```

### 核心参数
- `inputs`: 输入图片或文件夹路径（支持多个）
- `-o, --output`: 输出文件路径（.png或.pdf）
- `-c, --columns`: 列数布局（1-3列，默认自动）
- `--labels`: 添加字母编号 (a), (b), (c)...
- `-t, --title`: 图表标题
- `--matplotlib`: 使用matplotlib引擎（学术级）

### 完整参数列表
```
  -c {1,2,3}       列数布局（1-3列，默认自动判断）
  -t TITLE         图表标题
  -s SPACING       图片间距（像素，默认20）
  -m MARGIN        边距（像素，默认40）
  -f {png,pdf}     输出格式（默认根据扩展名判断）
  --matplotlib     使用matplotlib生成学术级图表
  --dpi DPI        输出分辨率（DPI，默认300）
  --labels         在每张图片上添加字母编号
  --label-position {top,bottom}  标签位置（默认top）
  --label-start N  起始编号偏移（0=a, 1=b, 2=c...，默认0）
```

## 使用示例

### 1. 基础拼接
```bash
# 文件夹内所有PNG自动拼接
python image_combiner.py /path/to/images/ -o combined.pdf
```

### 2. 学术图表（推荐）
```bash
# 带编号的专业图表
python image_combiner.py img1.png img2.png img3.png img4.png \
  -o results.pdf -c 2 --labels -t "Experimental Results"
```

### 3. 混合输入
```bash
# 文件夹+单个文件，三列布局
python image_combiner.py /folder1/ /folder2/ single.png \
  -o output.pdf -c 3 --labels -t "Comparison Study"
```

### 4. matplotlib版本
```bash
# 使用matplotlib引擎（更适合学术发表）
python image_combiner.py images/ -o academic.pdf \
  --matplotlib --labels -t "Research Findings"
```

### 5. 自定义标签功能 🆕
```bash
# 标签在底部显示
python image_combiner.py img1.png img2.png img3.png img4.png \
  -o bottom_labels.pdf -c 2 --labels --label-position bottom

# 从c开始编号（顶部）
python image_combiner.py *.png -o start_from_c.png \
  -c 2 --labels --label-start 2

# 底部标签，从e开始编号
python image_combiner.py images/ -o custom_labels.pdf \
  -c 3 --labels --label-position bottom --label-start 4 \
  -t "Custom Label Example"
```

### 6. 简洁输出
```bash
# 无标题、无编号的简洁拼接
python image_combiner.py *.png -o simple.png -c 2
```

## 输出格式说明

### PDF输出
- **优势**: 矢量图形，无损缩放，适合印刷
- **适用**: 学术论文、正式报告、期刊投稿
- **质量**: 高分辨率，专业级输出

### PNG输出
- **优势**: 广泛兼容，网页友好
- **适用**: 演示文稿、网页显示、社交媒体
- **质量**: 300 DPI，清晰锐利

## 布局说明

### 自动布局规则
- 1-2张图片：横向排列
- 3-6张图片：2列布局
- 7+张图片：3列布局

### 手动布局
- `-c 1`: 单列纵向排列
- `-c 2`: 双列网格
- `-c 3`: 三列网格

### 字母编号 🆕
- **顺序**: 从左到右，从上到下
- **格式**: (a), (b), (c), (d)...
- **位置**: 可选择图片上方或下方
- **起始**: 可自定义起始字母（如从c开始）
- **样式**: 白色背景框，清晰易读

## 间距优化

### v2.0 改进
- **问题解决**: 修复了标签与下方图片接触的问题
- **行间距**: 带标签时额外增加40像素间距
- **标签位置**: 优化字母编号位置，避免重叠

### 间距参数
```bash
# 自定义间距
python image_combiner.py images/ -o output.pdf \
  -s 30 -m 50 --labels  # 30px图片间距，50px边距
```

## 测试工具

```bash
# 运行测试脚本
python analysis/scripts/test_image_combiner.py

# 查看工具帮助
python analysis/scripts/test_image_combiner.py --help
```

## 最佳实践

### 学术论文使用
1. **格式选择**: 使用PDF格式确保印刷质量
2. **编号启用**: 启用 `--labels` 便于文中引用
3. **标题规范**: 使用英文标题，简洁明确
4. **布局选择**: 根据图片数量选择合适列数

```bash
# 学术论文标准配置
python image_combiner.py results/*.png \
  -o "Figure_1_Experimental_Results.pdf" \
  -c 2 --labels --matplotlib \
  -t "Comparison of Different Layout Strategies"
```

### 演示文稿使用
1. **格式选择**: PNG格式兼容性更好
2. **简洁设计**: 考虑是否需要标题和编号
3. **尺寸适配**: 根据演示需求调整布局

```bash
# 演示文稿配置
python image_combiner.py charts/*.png \
  -o presentation_slide.png \
  -c 3 --labels -t "Key Findings"
```

## 故障排除

### 常见问题

1. **字体警告**: 系统字体缺失，使用默认字体（不影响功能）
2. **内存不足**: 大图片数量过多，建议分批处理
3. **格式不支持**: 仅支持PNG输入，请转换其他格式

### 调试模式
```bash
# 查看详细输出
python image_combiner.py images/ -o debug.pdf --labels -v
```

## 更新日志

### v2.1 (最新版本) 🆕
- ✅ 添加自定义标签位置功能 `--label-position top|bottom`
- ✅ 添加自定义起始编号功能 `--label-start N`
- ✅ 支持标签功能组合使用（位置+起始编号）
- ✅ 完善matplotlib版本的标签支持
- ✅ 增强用户文档和测试脚本

### v2.0
- ✅ 移除图片文件名显示
- ✅ 添加字母编号功能 `--labels`
- ✅ 优化行间距，解决标签重叠问题
- ✅ 改进matplotlib渲染质量
- ✅ 增强错误处理和用户反馈

### v1.0
- ✅ 基础拼接功能
- ✅ 多种布局支持
- ✅ PDF/PNG输出
- ✅ 双引擎支持

## 技术支持

如有问题或建议，请检查：
1. 输入图片格式是否为PNG
2. 输出目录是否有写入权限
3. Python环境是否包含所需依赖

## 许可证

本工具为学术研究项目的一部分，遵循MIT许可证。

# 使用默认的24px标签（比原来大50%）
python image_combiner.py *.png -o default.png --labels

# 设置大标签（32px），适合高分辨率
python image_combiner.py *.png -o big_labels.png --labels --label-font-size 32 --v-spacing 60

# 设置小标签（18px），紧凑布局
python image_combiner.py *.png -o small_labels.png --labels --label-font-size 18 --v-spacing 40

# 超大标签用于打印/演示
python image_combiner.py *.png -o huge_labels.png --labels     --v-spacing 80