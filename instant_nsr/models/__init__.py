models = {} # ← 全局注册表，存储所有可用的模型类


def register(name):
    def decorator(cls):
        models[name] = cls  # ← 将类注册到字典中
        return cls
    return decorator


def make(name, config):
    """工厂模式：根据名字创建模型实例


    Args:
        name (_type_): _model 名字
        config (_type_): _model 配置

    Returns:
        _type_: _model 实例
    """
    model = models[name](config)    # ← 从字典查找类并实例化
    return model


from . import nerf, neus, geometry, texture
