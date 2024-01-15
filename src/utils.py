# '''
# Author: Wuyao 1955416359@qq.com
# Date: 2023-10-06 20:19:28
# LastEditors: Wuyao 1955416359@qq.com
# LastEditTime: 2023-11-21 21:26:01
# FilePath: \UnetV3\src\utils.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# '''
from PIL import Image

# H*W
def keep_image_size_open(path, size=(320, 240)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
def keep_image_size_open_rgb(path, size=(320, 240)):
    img = Image.open(path)
    # print(type(img))
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask
