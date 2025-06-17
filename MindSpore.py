import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
import hashlib
import random
from phe import paillier  # 同态加密库
from typing import List, Dict, Any, Tuple
import re  # 添加正则表达式支持

# 设置GPU上下文
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="GPU")

class AdvancedPrivacyPreserver:
    def __init__(self, num_parties: int, field_specs: List[Dict[str, Any]]):
        """
        高级隐私保护计算框架
        
        参数:
            num_parties: 参与方数量
            field_specs: 字段规范列表，每个字段指定:
                - name: 字段名
                - dtype: 数据类型 ('int', 'float', 'str')
                - is_sensitive: 是否敏感字段
                - dp_params: 差分隐私参数 (epsilon, delta)
                - needs_aggregation: 是否需要聚合
        """
        self.num_parties = num_parties
        self.field_specs = field_specs
        
        # 初始化同态加密密钥对
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        
        # 生成共享密钥(简化版)
        self.shared_keys = self._generate_shared_keys()
        
        # 创建GPU加速的随机数生成器
        self.rand = ops.UniformReal()
        self.laplace = ops.Laplace(seed=42)
        self.normal = ops.StandardNormal(seed=42)
    
    def _generate_shared_keys(self) -> Dict[Tuple[int, int], Any]:
        """生成参与方之间的共享密钥"""
        keys = {}
        for i in range(self.num_parties):
            for j in range(i+1, self.num_parties):
                # 为每个数值字段生成加密密钥
                key = {
                    'int': random.getrandbits(64),
                    'float': random.random()
                }
                keys[(i, j)] = key
        return keys
    
    # ================== GPU加速的差分隐私方法 ==================
    def _laplace_mechanism(self, value: float, sensitivity: float, epsilon: float) -> float:
        """GPU加速的拉普拉斯机制实现"""
        scale = sensitivity / epsilon
        # 使用MindSpore算子生成拉普拉斯噪声
        noise = self.laplace((1,)) * scale
        return float(value) + float(noise.asnumpy()[0])
    
    def _gaussian_mechanism(self, value: float, sensitivity: float, epsilon: float, delta: float) -> float:
        """GPU加速的高斯机制实现"""
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        # 使用MindSpore算子生成高斯噪声
        noise = self.normal((1,)) * sigma
        return float(value) + float(noise.asnumpy()[0])
    
    def apply_dp(self, value: Any, field_idx: int) -> Any:
        """应用差分隐私保护"""
        spec = self.field_specs[field_idx]
        if not spec.get('dp_params') or spec['dtype'] not in ['int', 'float']:
            return value
        
        epsilon, delta = spec['dp_params']['epsilon'], spec['dp_params'].get('delta', 1e-5)
        sensitivity = spec['dp_params'].get('sensitivity', 1.0)
        
        if delta > 0:
            return self._gaussian_mechanism(float(value), sensitivity, epsilon, delta)
        else:
            return self._laplace_mechanism(float(value), sensitivity, epsilon)
    
    # ================== 同态加密方法 ==================
    def homomorphic_encrypt(self, value: Any, field_idx: int) -> Any:
        """同态加密数据"""
        spec = self.field_specs[field_idx]
        if spec['dtype'] not in ['int', 'float'] or not spec.get('needs_aggregation'):
            return value
        
        if spec['dtype'] == 'int':
            return self.public_key.encrypt(int(value))
        else:
            return self.public_key.encrypt(float(value))
    
    def homomorphic_decrypt(self, encrypted: Any) -> Any:
        """同态解密数据"""
        if isinstance(encrypted, paillier.EncryptedNumber):
            return self.private_key.decrypt(encrypted)
        return encrypted
    
    def homomorphic_sum(self, encrypted_values: List[Any]) -> Any:
        """同态加密求和"""
        if not encrypted_values:
            return 0
        
        result = encrypted_values[0]
        for val in encrypted_values[1:]:
            if isinstance(val, paillier.EncryptedNumber):
                result += val
            else:
                result += val
        return result
    
    # ================== 数据脱敏方法 ==================
    def desensitize_string(self, value: str, pattern: str = None) -> str:
        """通用字符串脱敏"""
        if not pattern:
            if '@' in value:  # 邮箱
                parts = value.split('@')
                return f"{parts[0][0]}***@{parts[1]}"
            elif value.isdigit() and len(value) >= 7:  # 手机号/ID
                return f"{value[:3]}****{value[-4:]}"
            else:  # 姓名等
                return f"{value[0]}***" if len(value) > 1 else value
        else:
            # 使用正则表达式自定义脱敏
            return re.sub(pattern, '***', value)
    
    # ================== 核心处理方法 ==================
    def process_record(self, record: List[Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        处理单个记录，返回(脱敏数据, 加密数据)
        """
        desensitized = []
        encrypted = {}
        
        for i, (value, spec) in enumerate(zip(record, self.field_specs)):
            # 类型转换
            if spec['dtype'] == 'int':
                value = int(value)
            elif spec['dtype'] == 'float':
                value = float(value)
            else:
                value = str(value)
            
            # 数据脱敏
            if spec['is_sensitive'] and spec['dtype'] == 'str':
                desen_value = self.desensitize_string(value)
            else:
                desen_value = value
            
            # 应用差分隐私
            if spec.get('dp_params'):
                desen_value = self.apply_dp(desen_value, i)
            
            desensitized.append(desen_value)
            
            # 同态加密需要聚合的数值字段
            if spec.get('needs_aggregation') and spec['dtype'] in ['int', 'float']:
                encrypted[f'field_{i}'] = self.homomorphic_encrypt(value, i)
        
        return desensitized, encrypted
    
    def secure_aggregate(self, encrypted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """安全聚合加密数据"""
        aggregated = {}
        field_indices = [i for i, spec in enumerate(self.field_specs) 
                        if spec.get('needs_aggregation')]
        
        for idx in field_indices:
            key = f'field_{idx}'
            values = [party_data[key] for party_data in encrypted_data 
                    if key in party_data]
            
            # 同态求和
            sum_encrypted = self.homomorphic_sum(values)
            
            # 解密并应用差分隐私(如果需要)
            sum_decrypted = self.homomorphic_decrypt(sum_encrypted)
            spec = self.field_specs[idx]
            if spec.get('dp_params'):
                sum_decrypted = self.apply_dp(sum_decrypted, idx)
            
            aggregated[key] = {
                'sum': sum_decrypted,
                'average': sum_decrypted / len(encrypted_data),
                'count': len(values)
            }
        
        return aggregated


if __name__ == "__main__":
    # 设置GPU上下文
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="GPU")
    
    # 定义数据规范
    field_specs = [
        {'name': 'id', 'dtype': 'int', 'is_sensitive': False, 'needs_aggregation': False},
        {'name': 'name', 'dtype': 'str', 'is_sensitive': True, 'needs_aggregation': False},
        {'name': 'gender', 'dtype': 'str', 'is_sensitive': False, 'needs_aggregation': True},
        {'name': 'age', 'dtype': 'int', 'is_sensitive': True, 
         'needs_aggregation': True, 'dp_params': {'epsilon': 0.5, 'sensitivity': 1}},
        {'name': 'salary', 'dtype': 'float', 'is_sensitive': True,
         'needs_aggregation': True, 'dp_params': {'epsilon': 0.1, 'delta': 1e-5, 'sensitivity': 1000}},
        {'name': 'phone', 'dtype': 'str', 'is_sensitive': True, 'needs_aggregation': False}
    ]
    
    # 初始化隐私保护器(5个参与方)
    app = AdvancedPrivacyPreserver(num_parties=5, field_specs=field_specs)
    
    # 模拟输入数据
    party_inputs = [
        [101, "张三", "男", 32, 12500.00, "13012345678"],
        [102, "李四", "女", 28, 11800.00, "13087654321"],
        [103, "王五", "男", 45, 9000.00, "13511223344"],
        [104, "赵六", "女", 20, 6000.00, "13544332211"],
        [105, "孙七", "男", 27, 25000.00, "13988888888"],
    ]
    
    print("原始数据:")
    for i, data in enumerate(party_inputs):
        print(f"参与方 {i}: {data}")
    
    # 处理数据
    processed_data = []
    encrypted_data = []
    for record in party_inputs:
        desensitized, encrypted = app.process_record(record)
        processed_data.append(desensitized)
        encrypted_data.append(encrypted)
    
    print("\n脱敏后的数据:")
    for i, data in enumerate(processed_data):
        print(f"参与方 {i}: {data}")
    
    # 安全聚合
    aggregated = app.secure_aggregate(encrypted_data)
    
    print("\n安全聚合结果:")
    for field, stats in aggregated.items():
        idx = int(field.split('_')[1])
        field_name = field_specs[idx]['name']
        print(f"\n字段 '{field_name}' 统计:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    
    # 显示同态加密示例
    print("\n同态加密示例:")
    sample_value = 1000
    encrypted = app.homomorphic_encrypt(sample_value, 3)
    print(f"原始值: {sample_value} → 加密值: {encrypted.ciphertext()}...")
    decrypted = app.homomorphic_decrypt(encrypted)
    print(f"解密值: {decrypted}")
    
    # 显示差分隐私应用示例
    print("\n差分隐私应用示例:")
    # 年龄字段
    age_idx = 3
    original_age = 32
    dp_age = app.apply_dp(original_age, age_idx)
    print(f"年龄字段 - 原始值: {original_age}, 差分隐私处理后: {dp_age}")
    
    # 薪资字段
    salary_idx = 4
    original_salary = 12500.50
    dp_salary = app.apply_dp(original_salary, salary_idx)
    print(f"薪资字段 - 原始值: {original_salary}, 差分隐私处理后: {dp_salary}")