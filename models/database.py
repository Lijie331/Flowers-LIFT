"""
数据库模块 - 统一管理数据库连接
"""

import pymysql
from pymysql.cursors import DictCursor
import sys
import os

#获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DB_CONFIG

# 花卉信息缓存
CLASSNAMES = []  # 英文名称列表
CLASSNAMES_CN = {}  # 英文->中文对照表
CLASS_INFO = {}  # 完整花卉信息字典 {id: {name_en, name_cn}}


def get_db_connection():
    """获取数据库连接"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        print(f"database is connected~")
        return conn
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return None


def get_db_cursor(conn=None):
    """获取数据库游标（带自动连接管理）"""
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True
    
    if conn is None:
        return None, should_close
    
    cursor = conn.cursor(DictCursor)
    return cursor, should_close, conn


def execute_query(sql, params=None, fetch_one=False):
    """执行查询并返回结果"""
    conn = get_db_connection()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor(DictCursor)
        cursor.execute(sql, params)
        
        if fetch_one:
            result = cursor.fetchone()
        else:
            result = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        print(f"查询执行失败: {e}")
        if conn:
            conn.close()
        return None


def execute_update(sql, params=None):
    """执行更新操作（INSERT/UPDATE/DELETE）"""
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        affected = cursor.execute(sql, params)
        conn.commit()
        cursor.close()
        conn.close()
        return affected
    except Exception as e:
        print(f"更新执行失败: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False


# def load_flower_classes():
#     """从JSON文件加载花卉类别信息"""
#     global CLASSNAMES, CLASSNAMES_CN, CLASS_INFO
    
#     import json
#     from config import FLOWER_CLASSES_FILE
    
#     try:
#         with open(FLOWER_CLASSES_FILE, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         sorted_classes = sorted(data['classes'], key=lambda x: x['id'])
        
#         CLASSNAMES = [item['name_en'] for item in sorted_classes]
#         CLASSNAMES_CN = {item['name_en']: item['name_cn'] for item in sorted_classes}
#         CLASS_INFO = {item['id']: {'name_en': item['name_en'], 'name_cn': item['name_cn']} 
#                       for item in sorted_classes}
        
#         print(f"[INFO] 成功加载 {len(CLASSNAMES)} 个花卉类别")
#         return True
#     except Exception as e:
#         print(f"[ERROR] 加载花卉类别失败: {e}")
#         return False

def load_flower_classes():
    """从JSON文件加载花卉类别信息"""
    global CLASSNAMES, CLASSNAMES_CN, CLASS_INFO
    
    import json
    from config import FLOWER_CLASSES_FILE
    
    try:
        with open(FLOWER_CLASSES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 你的 JSON 结构中，花卉数据在 'flowers' 键下
        flowers_data = data.get('flowers', [])
        
        if not flowers_data:
            print(f"[ERROR] 未找到 'flowers' 键或数据为空")
            return False
        
        # 按 id 排序
        sorted_classes = sorted(flowers_data, key=lambda x: x.get('id', 0))
        
        # 根据你的 JSON 字段名调整
        CLASSNAMES = [item['latin_name'] for item in sorted_classes]  # 拉丁名作为英文名
        CLASSNAMES_CN = {item['latin_name']: item['chinese_name'] for item in sorted_classes}
        CLASS_INFO = {
            item['id']: {
                'name_en': item['latin_name'], 
                'name_cn': item['chinese_name']
            } 
            for item in sorted_classes
        }
        
        print(f"[INFO] 成功加载 {len(CLASSNAMES)} 个花卉类别")
        print(f"[INFO] 示例 - 拉丁名: {CLASSNAMES[0]}, 中文名: {CLASSNAMES_CN[CLASSNAMES[0]]}")
        return True
    except Exception as e:
        print(f"[ERROR] 加载花卉类别失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# 中文到英文映射缓存
CHINESE_TO_ENGLISH = {}


def init_chinese_to_english_mapping():
    """初始化中文到英文花卉名称映射"""
    global CHINESE_TO_ENGLISH
    CHINESE_TO_ENGLISH = {}
    
    for en_name, cn_name in CLASSNAMES_CN.items():
        CHINESE_TO_ENGLISH[cn_name] = en_name
        CHINESE_TO_ENGLISH[en_name] = en_name


def get_flower_folder_name(name):
    """获取花卉名称对应的文件夹名（中文名）"""
    # 如果名称是中文名，通过反向查找英文名
    if name in CLASSNAMES_CN.values():
        return name
    # 如果是英文名，返回中文名
    if name in CLASSNAMES_CN:
        return CLASSNAMES_CN[name]
    # 尝试模糊匹配
    for en_name, cn_name in CLASSNAMES_CN.items():
        if name.lower() in en_name.lower() or name.lower() in cn_name.lower():
            return cn_name
    return name


# 启动时加载花卉信息
load_flower_classes()
init_chinese_to_english_mapping()
