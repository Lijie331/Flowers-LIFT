"""
models包初始化文件
"""
import os
import torch
import sys

from .database import (
    get_db_connection,
    get_db_cursor,
    execute_query,
    execute_update,
    CLASSNAMES,
    CLASSNAMES_CN,
    CLASS_INFO,
    load_flower_classes,
    get_flower_folder_name,
)

from .models import PeftModelFromCLIP

# CLIP模型缓存路径
CLIP_CACHE_DIR = r'D:\1B.毕业设计\CLIP_cache'
os.environ['CLIP_CACHE_DIR'] = CLIP_CACHE_DIR


class ModelConfig:
    """模型配置类，用于替代argparse.Namespace"""
    def __init__(self, args=None):
        if args is not None:
            # 从args复制属性
            for key in dir(args):
                if not key.startswith('_'):
                    setattr(self, key, getattr(args, key))
        
        # 默认值设置
        if not hasattr(self, 'backbone'):
            self.backbone = 'CLIP-RN50'
        if not hasattr(self, 'classifier'):
            self.classifier = 'CosineClassifier'
        if not hasattr(self, 'scale'):
            self.scale = 25.0
        if not hasattr(self, 'init_style'):
            self.init_style = 'text_feat'
        if not hasattr(self, 'bias'):
            self.bias = 'none'
        if not hasattr(self, 'full_tuning'):
            self.full_tuning = False
        if not hasattr(self, 'bias_tuning'):
            self.bias_tuning = False
        if not hasattr(self, 'bn_tuning'):
            self.bn_tuning = False
        if not hasattr(self, 'ln_tuning'):
            self.ln_tuning = False
        if not hasattr(self, 'adapter'):
            self.adapter = False
        if not hasattr(self, 'lora'):
            self.lora = False
        if not hasattr(self, 'vpt_shallow'):
            self.vpt_shallow = False
        if not hasattr(self, 'vpt_deep'):
            self.vpt_deep = False
        if not hasattr(self, 'partial'):
            self.partial = None
        if not hasattr(self, 'vpt_len'):
            self.vpt_len = None
        if not hasattr(self, 'adapter_dim'):
            self.adapter_dim = 64
        if not hasattr(self, 'mask'):
            self.mask = False
        if not hasattr(self, 'mask_ratio'):
            self.mask_ratio = 0.0
        if not hasattr(self, 'mask_seed'):
            self.mask_seed = 42
        if not hasattr(self, 'ssf_attn'):
            self.ssf_attn = False
        if not hasattr(self, 'ssf_mlp'):
            self.ssf_mlp = False
        if not hasattr(self, 'ssf_ln'):
            self.ssf_ln = False
        if not hasattr(self, 'lora_mlp'):
            self.lora_mlp = False
        if not hasattr(self, 'adaptformer'):
            self.adaptformer = False


def _load_clip_model(model_name):
    """
    加载CLIP模型
    
    Args:
        model_name: 模型名称，如 'clip_rn50', 'clip_rn101', 'clip_vit_b_16'
    
    Returns:
        clip_model: CLIP模型
    """
    # 添加项目根目录到路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from clip import clip
        
        # 映射模型名称
        model_name_map = {
            'clip_rn50': 'RN50',
            'clip_rn101': 'RN101',
            'clip_vit_b_16': 'ViT-B/16',
            'clip_vit_b_32': 'ViT-B/32',
            'clip_vit_l_14': 'ViT-L/14',
        }
        
        if model_name not in model_name_map:
            model_name = 'clip_rn50'
        
        model_id = model_name_map[model_name]
        
        # 使用clip.load加载模型
        print(f'[INFO] 正在加载CLIP模型: {model_id}')
        clip_model, preprocess = clip.load(model_id, device='cpu')
        clip_model.float()
        
        print(f'[INFO] CLIP模型加载成功: {model_id}')
        return clip_model
        
    except Exception as e:
        print(f"[WARNING] CLIP模型加载失败: {e}")
        print(f"[INFO] 将使用简单的CNN模型替代")
        return None


def build_model(args, num_classes):
    """
    构建模型
    
    Args:
        args: 参数对象
        num_classes: 类别数量
    
    Returns:
        model: 训练好的模型
    """
    model_name = getattr(args, 'model', 'clip_rn50')
    
    # 尝试加载CLIP模型
    clip_model = _load_clip_model(model_name)
    
    if clip_model is not None:
        # 使用CLIP模型
        config = ModelConfig(args)
        
        # 设置backbone名称
        if 'rn50' in model_name.lower():
            config.backbone = 'CLIP-RN50'
        elif 'rn101' in model_name.lower():
            config.backbone = 'CLIP-RN101'
        elif 'vit_b_16' in model_name.lower():
            config.backbone = 'CLIP-ViT-B/16'
        elif 'vit_b_32' in model_name.lower():
            config.backbone = 'CLIP-ViT-B/32'
        elif 'vit_l_14' in model_name.lower():
            config.backbone = 'CLIP-ViT-L/14'
        
        model = PeftModelFromCLIP(config, clip_model, num_classes)
        print(f'[INFO] 使用CLIP模型: {config.backbone}')
        
    else:
        # 使用简单的CNN模型作为fallback
        model = _build_simple_cnn(num_classes)
        print(f'[INFO] 使用简单CNN模型')
    
    return model


def _build_simple_cnn(num_classes):
    """构建简单的CNN模型"""
    import torch.nn as nn
    
    model = nn.Sequential(
        # 基础卷积层
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model


__all__ = [
    'get_db_connection',
    'get_db_cursor',
    'execute_query',
    'execute_update',
    'CLASSNAMES',
    'CLASSNAMES_CN',
    'CLASS_INFO',
    'load_flower_classes',
    'get_flower_folder_name',
    'PeftModelFromCLIP',
    'build_model',
    'ModelConfig',
]
