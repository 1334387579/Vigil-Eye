# VigilEye - 智能驾驶员疲劳监测系统

![Demo](assets/demo.gif)

## 功能特性
✅ 实时眼部状态检测  
✅ 哈欠频率分析  
✅ 头部姿态估计  
✅ 违规行为识别（手机/饮食）  
✅ 多语言警告系统  

## 快速开始
```bash
git clone https://github.com/yourname/vigileye-dms.git
cd vigileye-dms
pip install -r requirements.txt
python main.py
```

## 技术架构
```mermaid
graph TD
    A[摄像头输入] --> B(MediaPipe面部检测)
    B --> C[眼部关键点分析]
    B --> D[嘴部状态识别]
    B --> E[头部姿态估计]
    A --> F[YOLOv10物体检测]
    C & D & E & F --> G[疲劳状态决策]
    G --> H[可视化预警界面]
```
## 贡献指南
欢迎提交 PR，请确保：
- 通过 `pylint` 检查（评分≥8.0）
- 添加相应的单元测试
- 更新文档说明
