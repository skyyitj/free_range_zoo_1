import pickle
import pprint

# 读取pickle文件
file_path = '/Users/theone/PycharmProjects/free_range_zoo_1/archive/competition_configs/wildfire/WS1.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        
    # 使用pprint美化输出
    print("Pickle文件内容：")
    pprint.pprint(data)
    
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
except pickle.UnpicklingError:
    print("错误：无法解析pickle文件")
except Exception as e:
    print(f"发生错误：{str(e)}")