import xml.etree.ElementTree as ET

def load_vehicle_config():
    """
    从vehicles.add.xml文件中读取车辆配置参数
    统一的配置加载函数，供所有脚本使用
    """
    vehicles_file = "data/vehicles.add.xml"
    config = {
        # 电池相关参数
        "capacity": 30000,  # 最大电池容量 (Wh)
        "maximumChargeRate": 22000,       # 最大充电功率 (W)
        "maximumPower": 22000,            # 最大功率 (W)
        
        # 充电策略参数
        "needToChargeLevel": 0.1,         # 充电触发阈值
        "saturatedChargeLevel": 0.8,      # 充满阈值
        "emptyThreshold": 0.05,           # 空电阈值
        
        # 充电站参数
        "efficiency": 0.95,               # 充电效率
        "chargeDelay": 1.0,               # 充电延迟 (s)
        
        # 车辆物理参数
        "mass": 1600,                     # 车辆质量 (kg)
        "frontSurfaceArea": 2.6,          # 前表面积 (m²)
        "airDragCoefficient": 0.35,       # 空气阻力系数
        "propulsionEfficiency": 0.8,      # 推进效率
        "recuperationEfficiency": 0.8,    # 回收效率
    }
    
    try:
        tree = ET.parse(vehicles_file)
        root = tree.getroot()
        
        # 查找EV车辆类型
        for vtype in root.findall("vType"):
            if vtype.get("id") == "EV":
                # 读取vType标签的属性
                mass = vtype.get("mass")
                if mass:
                    config["mass"] = float(mass)
                
                # 读取所有参数
                for param in vtype.findall("param"):
                    key = param.get("key")
                    value = param.get("value")
                    
                    if key == "device.battery.capacity":
                        config["capacity"] = float(value)
                    elif key == "device.battery.maximumChargeRate":
                        config["maximumChargeRate"] = float(value)
                    elif key == "maximumPower":
                        config["maximumPower"] = float(value)
                    elif key == "device.stationfinder.needToChargeLevel":
                        config["needToChargeLevel"] = float(value)
                    elif key == "device.stationfinder.saturatedChargeLevel":
                        config["saturatedChargeLevel"] = float(value)
                    elif key == "device.stationfinder.emptyThreshold":
                        config["emptyThreshold"] = float(value)
                    elif key == "mass":
                        config["mass"] = float(value)
                    elif key == "frontSurfaceArea":
                        config["frontSurfaceArea"] = float(value)
                    elif key == "airDragCoefficient":
                        config["airDragCoefficient"] = float(value)
                    elif key == "propulsionEfficiency":
                        config["propulsionEfficiency"] = float(value)
                    elif key == "recuperationEfficiency":
                        config["recuperationEfficiency"] = float(value)
                
                break
                
        print(f"✅ 车辆配置加载成功: {config}")
        
    except Exception as e:
        print(f"⚠️ 车辆配置加载失败，使用默认值: {e}")
    
    return config

def get_battery_config():
    """
    获取电池相关配置
    """
    config = load_vehicle_config()
    return {
        "capacity": config["capacity"],
        "needToChargeLevel": config["needToChargeLevel"],
        "saturatedChargeLevel": config["saturatedChargeLevel"],
        "emptyThreshold": config["emptyThreshold"]
    }

def get_charging_config():
    """
    获取充电相关配置
    """
    config = load_vehicle_config()
    return {
        "maximumChargeRate": config["maximumChargeRate"],
        "maximumPower": config["maximumPower"],
        "efficiency": config["efficiency"],
        "chargeDelay": config["chargeDelay"]
    }

def get_vehicle_physics_config():
    """
    获取车辆物理参数配置
    """
    config = load_vehicle_config()
    return {
        "mass": config["mass"],
        "frontSurfaceArea": config["frontSurfaceArea"],
        "airDragCoefficient": config["airDragCoefficient"],
        "propulsionEfficiency": config["propulsionEfficiency"],
        "recuperationEfficiency": config["recuperationEfficiency"]
    } 