import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = '/Users/theone/PycharmProjects/free_range_zoo_1/free_range_zoo/outputs/main_ams/2025-05-15/average_results.csv'
data = pd.read_csv(file_path)

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(data['Iteration'], data['Max'], label='Max', marker='o')
plt.plot(data['Iteration'], data['Min'], label='Min', marker='o')
plt.plot(data['Iteration'], data['Mean'], label='Mean', marker='o')

# 添加标题和标签
plt.title('Rewards With Iteration')
plt.xlabel('Iteration')
plt.ylabel('Values')
plt.legend()
plt.grid()

# 显示图形
plt.tight_layout()
plt.show()