import os
import cv2

# 输入文件夹路径
input_folder = "./dataset/TAG/IMX/0/tmp/0/"

# 输出文件夹路径
output_folder = "./dataset/TAG/IMX/0/tmp/renamed/"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取文件夹中所有jpg文件
jpg_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# 初始化重命名计数器
counter = 55

# 遍历文件并重命名/调整大小
for file_name in jpg_files:
    # 构建新的文件名
    new_file_name = f"800-n-{counter}.jpg"

    # 读取图像
    image = cv2.imread(os.path.join(input_folder, file_name))

    # 调整图像大小为800x600
    image_resized = cv2.resize(image, (800, 600))

    # 保存重命名后的图像
    output_path = os.path.join(output_folder, new_file_name)
    cv2.imwrite(output_path, image_resized)

    # 增加重命名计数器
    counter += 1

print("文件重命名和调整大小完成")