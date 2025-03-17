import os
import re
import imageio.v2 as imageio


def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else float("inf")


def create_gif(image_folder, output_gif, duration=500):
    images = []
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    image_files = [
        f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)
    ]
    image_files.sort(key=extract_number)

    for file_name in image_files:
        file_path = os.path.join(image_folder, file_name)
        images.append(imageio.imread(file_path))
    print(f"共找到 {len(images)} 张图片")
    if images:
        imageio.mimsave(output_gif, images, duration=duration / 1000, loop = 0)
        print(f"GIF 已成功保存至 {output_gif}")
    else:
        print("未找到有效图片文件，请检查文件夹路径")


image_folder = input("请输入图片文件夹路径：")
output_gif = "output.gif"
create_gif(image_folder, output_gif, duration=500)
