import math
import numpy as np

# 这里我给定了点P和线段AB两端点的坐标
a = np.asarray([-1, 1, 0])
b = np.asarray([1, 1, 0])
p = np.asarray([0, 0, 0])
# 计算用到的向量
ab = b - a
ap = p - a
bp = p - b
# 计算投影长度，并做正则化处理
r = np.dot(ap, ab) / (np.linalg.norm(ab)) ** 2
# 分了三种情况
if 0 < r < 1:
    dis = math.sqrt((np.linalg.norm(ap)) ** 2 - (r * np.linalg.norm(ab)) ** 2)
elif r >= 1:
    dis = np.linalg.norm(bp)
else:
    dis = np.linalg.norm(ap)
print(dis)
