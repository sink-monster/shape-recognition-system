# -*- coding: utf-8 -*-
"""
====================================================
课程设计：智能形状与物体识别系统
====================================================
作者：欧逸飞 50号
功能：
1. 二维几何形状识别（圆形、矩形、三角形等）
2. 物体识别（使用YOLOv8深度学习模型）
3. 图形用户界面（Tkinter）
技术栈：
- Python
- OpenCV (图像处理)
- Ultralytics YOLOv8 (深度学习)
- Tkinter (图形界面)
- NumPy, PIL (辅助库)
====================================================
"""

# ========== 导入必要的库 ==========
import cv2  # OpenCV库，用于图像处理
import numpy as np  # NumPy库，用于数值计算
from tkinter import *  # Tkinter库，用于创建图形界面
from tkinter import filedialog, messagebox, ttk  # Tkinter的补充组件
from PIL import Image, ImageTk, ImageDraw, ImageFont  # PIL库，用于图像显示和绘制
import os  # 操作系统库，用于文件路径操作
import sys  # 系统库，用于系统相关操作


# ========== 类1：二维几何形状识别器 ==========
class ShapeRecognizer:
    """
    二维几何形状识别类
    功能：检测图片中的圆形、矩形、三角形、五边形、六边形等几何形状
    """
    def __init__(self):
        """初始化形状识别器"""
        self.img = None  # 当前处理的图片
        self.original_img = None  # 原始图片（备份）
    
    def load_image(self, filepath):
        """
        加载图片（支持中文路径）
        参数：filepath - 图片文件路径
        返回：加载的图片，如果失败返回None
        """
        try:
            # 使用二进制方式读取图片，支持中文路径
            with open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8)
            # 用OpenCV解码图片
            self.img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            # 保存原始图片副本
            self.original_img = self.img.copy()
            return self.img if self.img is not None else None
        except Exception as e:
            print(f"加载图片失败: {e}")
            return None
    
    def get_dominant_color(self, contour):
        """
        获取形状内部的主要颜色
        参数：contour - 形状的轮廓
        返回：(B, G, R) 格式的颜色元组
        """
        # 创建一个和原图一样大的黑色遮罩
        mask = np.zeros(self.original_img.shape[:2], np.uint8)
        # 在遮罩上绘制填充的轮廓（白色）
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # 将图片分解为B、G、R三个通道
        b, g, r = cv2.split(self.original_img)
        
        # 计算每个通道在轮廓区域内的平均值
        b_mean = cv2.mean(b, mask=mask)[0]
        g_mean = cv2.mean(g, mask=mask)[0]
        r_mean = cv2.mean(r, mask=mask)[0]
        
        return (int(b_mean), int(g_mean), int(r_mean))
    
    def rgb_to_color_name(self, rgb):
        """
        将RGB值转换为中文颜色名称
        参数：rgb - (B, G, R)格式的颜色元组
        返回：颜色名称字符串
        """
        b, g, r = rgb
        
        # 将颜色值归一化到 0-1 范围
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        # 计算最大、最小值
        max_val = max(r_norm, g_norm, b_norm)
        min_val = min(r_norm, g_norm, b_norm)
        
        # 计算亮度和饱和度
        intensity = (max_val + min_val) / 2
        saturation = 0 if max_val == min_val else (max_val - min_val) / (1 - abs(2 * intensity - 1))
        
        # 如果饱和度很低，判断为黑白灰
        if saturation < 0.15:
            if intensity > 0.9:
                return "白色"
            elif intensity < 0.2:
                return "黑色"
            else:
                return "灰色"
        
        # 根据RGB分量判断颜色
        if r_norm >= g_norm and r_norm >= b_norm:
            if g_norm >= b_norm:
                if r_norm > 0.7 and g_norm > 0.5 and b_norm < 0.3:
                    return "橙色"
                else:
                    return "红色"
            else:
                return "红色"
        elif g_norm >= r_norm and g_norm >= b_norm:
            if r_norm > 0.6 and b_norm < 0.3:
                return "黄色"
            else:
                return "绿色"
        elif b_norm >= r_norm and b_norm >= g_norm:
            if r_norm > 0.5:
                return "紫色"
            else:
                return "蓝色"
        
        # 特殊颜色判断
        if r_norm > 0.6 and b_norm > 0.6 and g_norm < 0.3:
            return "品红色"
        if g_norm > 0.6 and b_norm > 0.6 and r_norm < 0.3:
            return "青色"
        
        return "灰色"
    
    def detect_shapes(self, min_area=100, epsilon_factor=0.04):
        """
        检测图片中的二维几何形状（核心算法）
        参数：
            min_area - 最小检测面积，小于此值的形状会被忽略
            epsilon_factor - 多边形近似系数，值越大近似越粗糙
        返回：形状信息列表
        """
        # 如果没有加载图片，直接返回
        if self.original_img is None:
            return None
        
        # ========== 步骤1：图像预处理 ==========
        img = self.original_img.copy()
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊，减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 膨胀边缘，连接断裂的边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # ========== 步骤2：轮廓检测 ==========
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []  # 存储识别到的形状
        
        # ========== 步骤3：遍历每个轮廓，识别形状 ==========
        for contour in contours:
            # 计算轮廓面积，过滤掉太小的轮廓
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # 计算轮廓周长
            perimeter = cv2.arcLength(contour, True)
            # 多边形近似，减少轮廓点数
            approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
            vertices = len(approx)  # 近似后的顶点数
            
            # 获取轮廓的外接矩形，计算宽高比
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            # ========== 步骤4：根据顶点数判断形状 ==========
            shape_name = "未知"
            shape_color = (0, 255, 0)  # 默认绿色
            
            if vertices == 3:
                # 3个顶点 → 三角形
                shape_name = "三角形"
                shape_color = (0, 165, 255)  # 橙色
            elif vertices == 4:
                # 4个顶点 → 判断是正方形还是矩形
                if 0.85 <= aspect_ratio <= 1.15:
                    # 宽高比接近1 → 正方形
                    shape_name = "正方形"
                    shape_color = (255, 0, 0)  # 蓝色
                else:
                    # 否则是矩形
                    shape_name = "矩形"
                    shape_color = (255, 255, 0)  # 青色
            elif vertices == 5:
                # 5个顶点 → 五边形
                shape_name = "五边形"
                shape_color = (0, 255, 255)  # 黄色
            elif vertices == 6:
                # 6个顶点 → 六边形
                shape_name = "六边形"
                shape_color = (255, 0, 255)  # 品红色
            elif vertices > 6:
                # 顶点多 → 判断是圆形还是椭圆形
                (center_x, center_y), radius = cv2.minEnclosingCircle(approx)
                circle_area = np.pi * radius ** 2
                circularity = area / circle_area  # 圆度
                
                if 0.65 <= circularity <= 1.35:
                    # 圆度好 → 圆形
                    shape_name = "圆形"
                    shape_color = (0, 0, 255)  # 红色
                else:
                    # 否则是椭圆形
                    shape_name = "椭圆形"
                    shape_color = (128, 0, 128)  # 紫色
            else:
                shape_name = f"{vertices}边形"
                shape_color = (128, 128, 128)  # 灰色
            
            # ========== 步骤5：计算形状的中心点 ==========
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # ========== 步骤6：识别形状的颜色 ==========
            dominant_color = self.get_dominant_color(contour)
            color_name = self.rgb_to_color_name(dominant_color)
            
            # 保存形状信息
            shapes.append({
                'name': shape_name,
                'color_name': color_name,
                'dominant_color': dominant_color,
                'contour': contour,
                'vertices': vertices,
                'area': area,
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'shape_color': shape_color
            })
        
        return shapes
    
    def draw_shapes(self, img, shapes):
        """
        在图片上绘制识别结果（使用PIL绘制中文）
        参数：
            img - 原始图片
            shapes - 形状信息列表
        返回：绘制后的图片
        """
        # 转换图片格式（BGR→RGB）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试加载中文字体
        font_paths = ['simhei.ttf', 'msyh.ttc', 'simsun.ttc', 'msyhbd.ttc']
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 14)
                break
            except:
                continue
        
        # 如果没找到字体，使用默认字体
        if font is None:
            font = ImageFont.load_default()
        
        # 遍历每个形状进行绘制
        for shape in shapes:
            x, y, w, h = shape['bbox']
            shape_color = shape['shape_color']
            cv_color = (shape_color[2], shape_color[1], shape_color[0])
            
            # 先转回OpenCV格式绘制轮廓和矩形
            cv2_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # 绘制轮廓
            cv2.drawContours(cv2_img, [shape['contour']], -1, shape_color, 3)
            # 绘制外接矩形
            cv2.rectangle(cv2_img, (x, y), (x + w, y + h), shape_color, 2)
            # 绘制中心点
            cv2.circle(cv2_img, shape['center'], 5, shape_color, -1)
            
            # 再转回PIL格式绘制中文标签
            img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 准备标签文字（颜色 + 形状名）
            label = f"{shape['color_name']}{shape['name']}"
            
            # 计算文字尺寸
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 计算标签位置（在形状上方）
            label_x = x
            label_y = y - text_height - 12
            
            # 如果位置超出图片上方，放到形状下方
            if label_y < 0:
                label_y = y + h + 5
            
            # 绘制标签背景和文字
            draw.rectangle(
                [(label_x, label_y), (label_x + text_width, label_y + text_height)],
                fill=shape_color
            )
            draw.text((label_x, label_y), label, font=font, fill=(255, 255, 255))
        
        # 最终结果转回OpenCV格式
        result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return result_img


# ========== 类2：物体识别器（YOLOv8深度学习） ==========
class ObjectRecognizer:
    """
    物体识别类
    功能：使用YOLOv8深度学习模型识别物体
    """
    def __init__(self):
        """初始化物体识别器"""
        self.img = None  # 当前图片
        self.results = None  # 识别结果
        self.current_model = None  # 当前加载的模型类型
        self.model = None  # YOLO模型对象
    
    def load_image(self, filepath):
        """
        加载图片（与形状识别器类似，支持中文路径）
        """
        try:
            with open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8)
            self.img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return self.img if self.img is not None else None
        except Exception as e:
            print(f"加载图片失败: {e}")
            return None
    
    def recognize_with_ultralytics(self, model_type='large', conf_threshold=0.3, iou_threshold=0.45, use_tta=False):
        """
        使用 YOLOv8 进行高精度识别（核心算法）
        参数：
            model_type - 模型类型：nano, small, medium, large
            conf_threshold - 置信度阈值（0-1）
            iou_threshold - IOU阈值（0-1）
            use_tta - 是否使用测试时间增强
        返回：检测结果列表
        """
        try:
            # 动态导入ultralytics库（避免启动时就报错）
            from ultralytics import YOLO
            
            # 模型文件名映射
            model_map = {
                'nano': 'yolov8n.pt',    # 最小，最快
                'small': 'yolov8s.pt',   # 小，较快（推荐）
                'medium': 'yolov8m.pt',  # 中等
                'large': 'yolov8x.pt'    # 最大，最准确
            }
            
            model_name = model_map.get(model_type, 'yolov8s.pt')
            print(f"正在加载模型: {model_name}")
            
            # 如果模型类型改变，重新加载模型
            if self.current_model != model_type:
                self.model = YOLO(model_name)  # 首次运行会自动下载模型
                self.current_model = model_type
            
            # ========== 使用模型进行推理 ==========
            results = self.model(
                self.img,
                conf=conf_threshold,  # 置信度阈值
                iou=iou_threshold,    # IOU阈值
                augment=use_tta,      # 是否使用TTA
                verbose=False         # 不打印详细信息
            )
            
            # ========== 解析识别结果 ==========
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # 获取置信度
                        conf = box.conf[0].cpu().numpy()
                        # 获取类别ID
                        cls_id = int(box.cls[0].cpu().numpy())
                        # 获取类别名称
                        cls_name = self.model.names[cls_id]
                        
                        detections.append({
                            'class_name': cls_name,
                            'confidence': float(conf),
                            'box': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                        })
            
            # 按置信度从高到低排序
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            return detections
            
        except ImportError:
            print("ultralytics 未安装")
            return None
        except Exception as e:
            print(f"YOLOv8 识别失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def draw_results(self, img, detections):
        """
        绘制物体识别结果
        """
        img_copy = img.copy()
        
        # 不同物体使用不同颜色
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (192, 192, 192), (128, 128, 128), (255, 128, 0), (255, 0, 128)
        ]
        
        for i, det in enumerate(detections):
            x, y, w, h = det['box']
            label = f"{det['class_name']}: {det['confidence']:.2%}"
            color = colors[i % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 3)
            
            # 绘制标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # 绘制标签背景
            cv2.rectangle(
                img_copy,
                (x, y - text_height - 12),
                (x + text_width, y),
                color,
                -1
            )
            
            # 绘制文字
            cv2.putText(
                img_copy,
                label,
                (x, y - 6),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return img_copy


# ========== 类3：图形界面主程序 ==========
class RecognitionApp:
    """
    图形界面类
    功能：创建用户友好的图形界面
    """
    def __init__(self, root):
        """初始化图形界面"""
        self.root = root
        self.root.title("🎨 智能识别系统 - 形状 & 物体")
        self.root.geometry("1650x1050")  # 窗口大小
        self.root.configure(bg="#f8fafc")  # 更柔和的背景色
        
        # 创建识别器对象
        self.object_recognizer = ObjectRecognizer()
        self.shape_recognizer = ShapeRecognizer()
        
        self.current_img = None  # 当前显示的图片
        self.original_img = None  # 原始图片
        self.mode_var = StringVar(value="shape")  # 识别模式变量
        
        # 建立界面
        self.setup_ui()
    
    def setup_ui(self):
        """建立整个界面布局"""
        main_container = Frame(self.root, bg="#f8fafc")
        main_container.pack(fill=BOTH, expand=True, padx=30, pady=30)
        
        # ========== 左侧控制面板 ==========
        left_panel = Frame(main_container, width=480, bg="#ffffff", relief=FLAT, bd=0)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 25))
        left_panel.pack_propagate(False)
        
        # 添加更好的阴影效果
        shadow1 = Frame(left_panel, bg="#e2e8f0")
        shadow1.place(x=2, y=2, relwidth=1, relheight=1)
        shadow2 = Frame(left_panel, bg="#cbd5e1")
        shadow2.place(x=4, y=4, relwidth=1, relheight=1)
        
        content_frame = Frame(left_panel, bg="#ffffff")
        content_frame.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.create_control_panel(content_frame)
        
        # ========== 右侧显示区域 ==========
        right_panel = Frame(main_container, bg="#f8fafc")
        right_panel.pack(side=RIGHT, fill=BOTH, expand=True)
        
        self.create_canvas_area(right_panel)  # 图片显示区
        self.create_info_panel(right_panel)   # 结果显示区
    
    def create_control_panel(self, parent):
        """创建左侧控制面板"""
        # ========== 标题区 ==========
        header_frame = Frame(parent, bg="#e0f2fe", height=110)
        header_frame.pack(fill=X)
        header_frame.pack_propagate(False)
        
        # 标题装饰
        decor_frame = Frame(header_frame, bg="#e0f2fe")
        decor_frame.pack(expand=True, padx=25)
        
        title_icon = Label(decor_frame, text="🎯", 
                         font=("Arial", 38), 
                         bg="#e0f2fe", fg="#0369a1")
        title_icon.pack(side=LEFT, padx=(0, 18))
        
        title_label = Label(decor_frame, text="智能识别系统", 
                           font=("Microsoft YaHei", 24, "bold"), 
                           bg="#e0f2fe", fg="#0c4a6e")
        title_label.pack(side=LEFT)
        
        # ========== 模式选择区 ==========
        mode_container = Frame(parent, bg="#ffffff", padx=25, pady=25)
        mode_container.pack(fill=X)
        
        Label(mode_container, text="📌 识别模式", 
             font=("Microsoft YaHei", 14, "bold"), 
             bg="#ffffff", fg="#1e293b", anchor=W).pack(fill=X, pady=(0, 15))
        
        mode_buttons_frame = Frame(mode_container, bg="#ffffff")
        mode_buttons_frame.pack(fill=X)
        
        # 形状识别按钮 - 更美观的样式
        self.shape_btn = Button(mode_buttons_frame, text="🔷 二维形状识别", 
                               command=lambda: self.set_mode("shape"),
                               font=("Microsoft YaHei", 12, "bold"),
                               bg="#3b82f6", fg="white",
                               relief=RAISED, bd=0, padx=25, pady=16, cursor="hand2",
                               activebackground="#2563eb", activeforeground="white")
        self.shape_btn.pack(side=LEFT, fill=X, expand=True, padx=(0, 8))
        
        # 物体识别按钮 - 更美观的样式
        self.object_btn = Button(mode_buttons_frame, text="📦 物体识别", 
                               command=lambda: self.set_mode("object"),
                               font=("Microsoft YaHei", 12, "bold"),
                               bg="#f1f5f9", fg="#475569",
                               relief=RAISED, bd=0, padx=25, pady=16, cursor="hand2",
                               activebackground="#e2e8f0", activeforeground="#1e293b")
        self.object_btn.pack(side=LEFT, fill=X, expand=True, padx=(8, 0))
        
        # ========== 参数面板（可滚动） ==========
        params_scroll = Frame(parent, bg="#ffffff")
        params_scroll.pack(fill=BOTH, expand=True)
        
        canvas = Canvas(params_scroll, bg="#ffffff", highlightthickness=0)
        scrollbar = Scrollbar(params_scroll, orient=VERTICAL, command=canvas.yview)
        self.params_container = Frame(canvas, bg="#ffffff")
        
        self.params_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.params_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 添加鼠标滚轮滚动支持
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind("<Enter>", _bind_to_mousewheel)
        canvas.bind("<Leave>", _unbind_from_mousewheel)
        
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # 形状识别参数面板
        self.shape_params_frame = Frame(self.params_container, bg="#ffffff")
        self.shape_params_frame.pack(fill=X)
        self.create_shape_params(self.shape_params_frame)
        
        # 物体识别参数面板（初始隐藏）
        self.object_params_frame = Frame(self.params_container, bg="#ffffff")
        self.object_params_frame.pack(fill=X)
        self.object_params_frame.pack_forget()
        self.create_object_params(self.object_params_frame)
        
        # ========== 操作按钮区（固定在底部） ==========
        buttons_container = Frame(parent, bg="#ffffff", padx=25, pady=25)
        buttons_container.pack(fill=X, side=BOTTOM)
        
        # 添加优雅的分隔线
        separator = Frame(buttons_container, bg="#e2e8f0", height=2)
        separator.pack(fill=X, pady=(0, 20))
        
        # 加载图片按钮 - 更美观
        load_btn = Button(buttons_container, text="📂 加载图片", 
                         command=self.load_image,
                         font=("Microsoft YaHei", 12, "bold"),
                         bg="#10b981", fg="white",
                         relief=RAISED, bd=0, padx=25, pady=16, cursor="hand2",
                         activebackground="#059669", activeforeground="white")
        load_btn.pack(fill=X, pady=(0, 12))
        
        # 保存结果按钮 - 更美观
        save_btn = Button(buttons_container, text="💾 保存结果", 
                         command=self.save_image,
                         font=("Microsoft YaHei", 12, "bold"),
                         bg="#6366f1", fg="white",
                         relief=RAISED, bd=0, padx=25, pady=16, cursor="hand2",
                         activebackground="#4f46e5", activeforeground="white")
        save_btn.pack(fill=X, pady=(0, 12))
        
        # 开始识别按钮（更大更显眼，带渐变色风格）
        recognize_btn = Button(buttons_container, text="🚀 开始识别", 
                              command=self.recognize,
                              font=("Microsoft YaHei", 15, "bold"),
                              bg="#f59e0b", fg="white",
                              relief=RAISED, bd=0, padx=25, pady=20, cursor="hand2",
                              activebackground="#d97706", activeforeground="white")
        recognize_btn.pack(fill=X)
    
    def set_mode(self, mode):
        """设置识别模式"""
        self.mode_var.set(mode)
        self.on_mode_change()
    
    def create_shape_params(self, parent):
        """创建形状识别的参数滑块"""
        section_frame = Frame(parent, bg="#ffffff", padx=15, pady=20)
        section_frame.pack(fill=X)
        
        # 参数区域标题 - 更美观
        title_frame = Frame(section_frame, bg="#f8fafc")
        title_frame.pack(fill=X, pady=(0, 18))
        
        Label(title_frame, text="⚙️ 形状识别参数", 
             font=("Microsoft YaHei", 13, "bold"), 
             bg="#f8fafc", fg="#1e293b", anchor=W,
             padx=15, pady=12).pack(fill=X)
        
        # 最小检测面积滑块
        self.min_area_var = IntVar(value=100)
        self.create_param_slider(section_frame, "最小检测面积", self.min_area_var,
                                20, 1000, 20, "px²")
        
        # 多边形近似系数滑块
        self.epsilon_var = DoubleVar(value=0.04)
        self.create_param_slider(section_frame, "多边形近似系数", self.epsilon_var,
                                0.01, 0.1, 0.005, "", 3)
    
    def create_param_slider(self, parent, label, var, min_val, max_val, step, suffix="", decimals=0):
        """
        创建通用的参数滑块
        """
        frame = Frame(parent, bg="#ffffff", pady=15)
        frame.pack(fill=X)
        
        # 标签 - 更美观
        label_frame = Frame(frame, bg="#ffffff")
        label_frame.pack(fill=X, pady=(0, 8))
        
        Label(label_frame, text=label, bg="#ffffff", 
             font=("Microsoft YaHei", 11), fg="#334155", anchor=W).pack(side=LEFT)
        
        # 滑块和数值显示
        slider_frame = Frame(frame, bg="#ffffff")
        slider_frame.pack(fill=X)
        
        # 创建更美观的滑块
        slider = Scale(slider_frame, from_=min_val, to=max_val, resolution=step,
                      orient=HORIZONTAL, variable=var,
                      bg="#f8fafc", fg="#1e293b", highlightthickness=0,
                      troughcolor="#e2e8f0", width=14, length=240,
                      activebackground="#3b82f6")
        slider.pack(side=LEFT, fill=X, expand=True)
        
        # 显示当前值的函数
        def get_display_value():
            if decimals == 0:
                return f"{var.get()}{suffix}"
            else:
                return f"{var.get():.{decimals}f}{suffix}"
        
        # 数值显示框 - 更美观
        value_frame = Frame(slider_frame, bg="#3b82f6", relief=FLAT, bd=0)
        value_frame.pack(side=LEFT, padx=(18, 0))
        
        value_label = Label(value_frame, text=get_display_value(), 
                           bg="#3b82f6", fg="white",
                           font=("Microsoft YaHei", 11, "bold"),
                           padx=12, pady=5)
        value_label.pack()
        
        # 值变化时更新标签
        def update_label(*args):
            value_label.config(text=get_display_value())
        
        var.trace('w', update_label)
    
    def create_object_params(self, parent):
        """创建物体识别的参数面板"""
        section_frame = Frame(parent, bg="#ffffff", padx=15, pady=20)
        section_frame.pack(fill=X)
        
        # 参数区域标题 - 更美观
        title_frame = Frame(section_frame, bg="#f8fafc")
        title_frame.pack(fill=X, pady=(0, 18))
        
        Label(title_frame, text="🤖 物体识别参数", 
             font=("Microsoft YaHei", 13, "bold"), 
             bg="#f8fafc", fg="#1e293b", anchor=W,
             padx=15, pady=12).pack(fill=X)
        
        # 模型大小选择
        model_container = Frame(section_frame, bg="#ffffff")
        model_container.pack(fill=X, pady=(0, 10))
        
        Label(model_container, text="模型大小", 
             bg="#ffffff", font=("Microsoft YaHei", 9), 
             fg="#34495e").pack(anchor=W)
        
        model_grid = Frame(model_container, bg="#ffffff")
        model_grid.pack(fill=X, pady=(5, 0))
        
        self.model_var = StringVar(value='small')
        
        # 4个模型选项
        model_options = [
            ("nano", "Nano (最快)", "#95a5a6"),
            ("small", "Small (推荐)", "#27ae60"),
            ("medium", "Medium", "#f39c12"),
            ("large", "Large (最准)", "#e74c3c")
        ]
        
        for idx, (value, text, color) in enumerate(model_options):
            row = idx // 2
            col = idx % 2
            self.create_model_button(model_grid, value, text, color, row, col)
        
        # 置信度阈值滑块
        self.conf_var = DoubleVar(value=0.3)
        self.create_param_slider(section_frame, "置信度阈值", self.conf_var,
                                0.05, 0.8, 0.05, "%")
        
        # IOU阈值滑块
        self.iou_var = DoubleVar(value=0.45)
        self.create_param_slider(section_frame, "IOU阈值", self.iou_var,
                                0.1, 0.8, 0.05, "%")
        
        # TTA开关
        self.tta_var = BooleanVar(value=False)
        tta_container = Frame(section_frame, bg="#ffffff", pady=10)
        tta_container.pack(fill=X)
        
        tta_check = Checkbutton(tta_container, text="启用测试时间增强 (TTA)", 
                               variable=self.tta_var, bg="#ffffff",
                               font=("Microsoft YaHei", 9), fg="#34495e",
                               activebackground="#ffffff", selectcolor="#3498db")
        tta_check.pack(anchor=W)
    
    def create_model_button(self, parent, value, text, color, row, col):
        """创建模型选择按钮"""
        btn = Button(parent, text=text,
                   command=lambda: self.model_var.set(value),
                   font=("Microsoft YaHei", 10),
                   bg="#f1f5f9", fg="#475569",
                   relief=RAISED, bd=0, padx=15, pady=12, cursor="hand2",
                   activebackground="#e2e8f0", activeforeground="#1e293b")
        btn.grid(row=row, column=col, sticky="ew", padx=5, pady=5)
        parent.grid_columnconfigure(col, weight=1)
        
        # 更新按钮样式
        def update_btn_style(*args):
            if self.model_var.get() == value:
                btn.config(bg=color, fg="white")
            else:
                btn.config(bg="#e5e7eb", fg="#4b5563")
        
        self.model_var.trace('w', update_btn_style)
        update_btn_style()
    
    def on_mode_change(self):
        """模式切换时更新界面显示"""
        mode = self.mode_var.get()
        if mode == "shape":
            # 显示形状参数，隐藏物体参数
            self.shape_params_frame.pack(fill=X)
            self.object_params_frame.pack_forget()
            self.shape_btn.config(bg="#3498db", fg="white")
            self.object_btn.config(bg="#ecf0f1", fg="#2c3e50")
        else:
            # 显示物体参数，隐藏形状参数
            self.shape_params_frame.pack_forget()
            self.object_params_frame.pack(fill=X)
            self.shape_btn.config(bg="#ecf0f1", fg="#2c3e50")
            self.object_btn.config(bg="#3498db", fg="white")
    
    def create_canvas_area(self, parent):
        """创建图片显示区域"""
        canvas_container = Frame(parent, bg="#ffffff", relief=FLAT, bd=0)
        canvas_container.pack(fill=BOTH, expand=True)
        
        # 更好的阴影效果
        canvas_shadow1 = Frame(canvas_container, bg="#e2e8f0")
        canvas_shadow1.place(x=2, y=2, relwidth=1, relheight=1)
        canvas_shadow2 = Frame(canvas_container, bg="#cbd5e1")
        canvas_shadow2.place(x=4, y=4, relwidth=1, relheight=1)
        
        canvas_content = Frame(canvas_container, bg="#ffffff")
        canvas_content.place(x=0, y=0, relwidth=1, relheight=1)
        
        header = Frame(canvas_content, bg="#f0f9ff", height=55)
        header.pack(fill=X)
        header.pack_propagate(False)
        
        Label(header, text="🖼️ 图片预览", 
             font=("Microsoft YaHei", 14, "bold"), 
             bg="#f0f9ff", fg="#0c4a6e").pack(expand=True)
        
        self.canvas = Canvas(canvas_content, bg="#f8fafc", cursor="hand2")
        self.canvas.pack(fill=BOTH, expand=True, padx=15, pady=15)
        
        # 初始提示文字 - 更美观
        self.hint_text = self.canvas.create_text(450, 260, 
                                                 text="📂 点击「加载图片」开始识别", 
                                                 font=("Microsoft YaHei", 20), 
                                                 fill="#64748b")
    
    def create_info_panel(self, parent):
        """创建结果信息面板"""
        info_frame = Frame(parent, bg="#ffffff", height=260, relief=FLAT, bd=0)
        info_frame.pack(fill=X, pady=(25, 0))
        info_frame.pack_propagate(False)
        
        # 更好的阴影效果
        info_shadow1 = Frame(info_frame, bg="#e2e8f0")
        info_shadow1.place(x=2, y=2, relwidth=1, relheight=1)
        info_shadow2 = Frame(info_frame, bg="#cbd5e1")
        info_shadow2.place(x=4, y=4, relwidth=1, relheight=1)
        
        info_content = Frame(info_frame, bg="#ffffff")
        info_content.place(x=0, y=0, relwidth=1, relheight=1)
        
        result_header = Frame(info_content, bg="#ecfdf5", height=55)
        result_header.pack(fill=X)
        result_header.pack_propagate(False)
        
        Label(result_header, text="📋 识别结果", 
             font=("Microsoft YaHei", 14, "bold"), 
             bg="#ecfdf5", fg="#065f46").pack(expand=True)
        
        text_container = Frame(info_content, bg="#f0fdf4")
        text_container.pack(fill=BOTH, expand=True, padx=15, pady=15)
        
        self.result_text = Text(text_container, height=8, 
                               font=("Microsoft YaHei", 12),
                               bg="#ffffff", fg="#0f172a",
                               relief=FLAT, bd=0,
                               padx=18, pady=18)
        self.result_text.pack(side=LEFT, fill=BOTH, expand=True)
        
        scrollbar = Scrollbar(text_container, command=self.result_text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
    
    def load_image(self):
        """加载图片按钮的功能"""
        # 弹出文件选择对话框
        filepath = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*")
            ]
        )
        
        if filepath:
            # 同时加载到两个识别器
            img = self.object_recognizer.load_image(filepath)
            self.shape_recognizer.load_image(filepath)
            
            if img is not None:
                # 保存原始图片和当前显示图片
                self.original_img = img.copy()
                self.current_img = img.copy()
                # 更新结果面板
                self.result_text.delete(1.0, END)
                self.result_text.insert(END, f"已加载图片: {os.path.basename(filepath)}\n")
                self.result_text.insert(END, f"图片尺寸: {img.shape[1]}x{img.shape[0]}\n\n")
                self.result_text.insert(END, "请选择模式并点击\"开始识别\"\n")
                # 更新画布显示
                self.update_canvas()
            else:
                messagebox.showerror("错误", "无法加载图片")
    
    def save_image(self):
        """保存结果图片"""
        if self.current_img is None:
            messagebox.showwarning("警告", "没有可保存的图片")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="保存图片",
            defaultextension=".png",
            filetypes=[
                ("PNG 图片", "*.png"),
                ("JPEG 图片", "*.jpg"),
                ("所有文件", "*.*")
            ]
        )
        
        if filepath:
            try:
                ext = os.path.splitext(filepath)[1].lower()
                if ext == '.jpg' or ext == '.jpeg':
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    result, buf = cv2.imencode('.jpg', self.current_img, encode_param)
                else:
                    result, buf = cv2.imencode('.png', self.current_img)
                
                if result:
                    with open(filepath, 'wb') as f:
                        f.write(buf)
                    messagebox.showinfo("成功", f"图片已保存到:\n{filepath}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {e}")
    
    def recognize(self):
        """开始识别按钮的功能"""
        if self.original_img is None:
            messagebox.showwarning("警告", "请先加载图片")
            return
        
        mode = self.mode_var.get()
        
        # 根据当前模式选择识别方法
        if mode == "shape":
            self.recognize_shapes()
        else:
            self.recognize_objects()
    
    def recognize_shapes(self):
        """执行二维形状识别"""
        self.result_text.delete(1.0, END)
        self.result_text.insert(END, "正在进行二维形状识别...\n\n")
        self.root.update()
        
        # 获取参数
        min_area = self.min_area_var.get()
        epsilon = self.epsilon_var.get()
        
        # 调用识别函数
        shapes = self.shape_recognizer.detect_shapes(min_area, epsilon)
        
        if shapes:
            # 绘制结果
            self.current_img = self.shape_recognizer.draw_shapes(self.original_img, shapes)
            self.update_canvas()
            
            # 更新结果面板
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, f"✅ 识别到 {len(shapes)} 个二维形状:\n\n")
            
            for i, shape in enumerate(shapes, 1):
                self.result_text.insert(END, f"{i}. {shape['color_name']}{shape['name']}\n")
                self.result_text.insert(END, f"   面积: {int(shape['area'])} px²\n")
                self.result_text.insert(END, f"   顶点数: {shape['vertices']}\n")
                self.result_text.insert(END, f"   中心: {shape['center']}\n")
                self.result_text.insert(END, f"   RGB: {shape['dominant_color']}\n\n")
        else:
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, "未识别到任何二维形状\n\n")
            self.result_text.insert(END, "建议：降低最小检测面积或调整图片")
    
    def recognize_objects(self):
        """执行物体识别"""
        self.result_text.delete(1.0, END)
        
        # 获取参数
        model_type = self.model_var.get()
        conf_threshold = self.conf_var.get()
        iou_threshold = self.iou_var.get()
        use_tta = self.tta_var.get()
        
        self.result_text.insert(END, f"正在使用 {model_type.upper()} 模型进行物体识别...\n")
        self.result_text.insert(END, f"置信度: {int(conf_threshold*100)}%, IOU: {int(iou_threshold*100)}%\n")
        if use_tta:
            self.result_text.insert(END, "TTA: 已启用\n")
        self.result_text.insert(END, "\n")
        self.result_text.insert(END, "（首次使用新模型会自动下载，请耐心等待）\n\n")
        self.root.update()
        
        # 调用识别函数
        detections = self.object_recognizer.recognize_with_ultralytics(
            model_type=model_type,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            use_tta=use_tta
        )
        
        if detections is None:
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, "识别失败\n\n")
            messagebox.showerror("错误", "识别过程出错，请检查控制台信息")
            return
        
        if detections:
            # 绘制结果
            self.current_img = self.object_recognizer.draw_results(self.original_img, detections)
            self.update_canvas()
            
            # 更新结果面板
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, f"✅ 识别到 {len(detections)} 个物体:\n\n")
            
            for i, det in enumerate(detections, 1):
                self.result_text.insert(END, f"{i}. {det['class_name']}\n")
                self.result_text.insert(END, f"   置信度: {det['confidence']:.2%}\n\n")
        else:
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, "未识别到任何物体\n\n")
            self.result_text.insert(END, "建议：降低置信度阈值或尝试更大的模型")
    
    def update_canvas(self):
        """更新画布上的图片显示"""
        if self.current_img is not None:
            # 隐藏提示文字
            self.canvas.itemconfigure(self.hint_text, state='hidden')
            
            # 转换图片格式用于显示
            img_rgb = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # 缩放图片以适应画布
            if canvas_width > 1 and canvas_height > 1:
                img_width, img_height = img_pil.size
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_size = (int(img_width * scale), int(img_height * scale))
                img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
            
            self.photo_img = ImageTk.PhotoImage(img_pil)
            
            self.canvas.delete("all")
            
            # 居中显示图片
            x = (canvas_width - self.photo_img.width()) // 2
            y = (canvas_height - self.photo_img.height()) // 2
            
            self.canvas.create_image(x, y, image=self.photo_img, anchor=NW)


# ========== 主程序入口 ==========
def main():
    """
    程序主入口
    创建Tkinter窗口并启动事件循环
    """
    root = Tk()
    app = RecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
