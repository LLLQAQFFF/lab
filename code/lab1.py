import random
import numpy as np
import matplotlib.pyplot as plt

# hopfield 网络
class hopfield:
    # 初始化权重
    def __init__(self, Data):
        n = Data[0].shape[-1]
        k = len(Data)
        W = np.zeros((n, n))
        # 权重为数据的内积之和
        W = sum(np.matmul(2*Data[a].reshape(n, 1)-1, 2*Data[a].reshape(1, n)-1) for a in range(k))
        for i in range(n):
            W[i, i] = 0
        self.W = W / float(n)
        self.n = n
        self.k = k

    # 进行图像的联想预测
    def fit(self, inMat):
        newMat = np.zeros(len(inMat))
        # 异步更新，更新顺序为顺序
        while np.sum(np.abs(newMat - inMat)) != 0:
            newMat = np.copy(inMat)
            for i in range(self.n):
                temp = np.dot(inMat, self.W[:, i])
                if temp >= 0:
                    inMat[i] = 1
                else:
                    inMat[i] = -1
        return newMat


Data = [np.array([-1, 1, 1, 1, -1,
                  1, -1, -1, -1, 1,
                  1, -1, -1, -1, 1,
                  1, -1, -1, -1, 1,
                  1, -1, -1, -1, 1,
                  -1, 1, 1, 1, -1]),  # 0
        np.array([-1, 1, 1, -1, -1,
                  -1, -1, 1, -1, -1,
                  -1, -1, 1, -1, -1,
                  -1, -1, 1, -1, -1,
                  -1, -1, 1, -1, -1,
                  -1, -1, 1, -1, -1]),  # 1
        np.array([1, 1, 1, 1, 1,
                  -1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1,
                  1, -1, -1, -1, -1,
                  1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1]),  # 2
        np.array([1, 1, 1, 1, 1,
                  -1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1,
                  -1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1]),  # 3
        np.array([1, -1, 1, -1, -1,
                  1, -1, 1, -1, -1,
                  1, -1, 1, -1, -1,
                  1, 1, 1, 1, 1,
                  -1, -1, 1, -1, -1,
                  -1, -1, 1, -1, -1]),  # 4
        np.array([1, 1, 1, 1, 1,
                  1, -1, -1, -1, -1,
                  1, 1, 1, 1, 1,
                  -1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1]),  # 5
        np.array([1, 1, 1, 1, 1,
                  1, -1, -1, -1, -1,
                  1, 1, 1, 1, 1,
                  1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1]),  # 6
        np.array([1, 1, 1, 1, 1,
                  -1, -1, -1, -1, 1,
                  -1, -1, -1, 1, -1,
                  -1, -1, -1, 1, -1,
                  -1, -1, 1, -1, -1,
                  -1, 1, -1, -1, -1]),  # 7
        np.array([1, 1, 1, 1, 1,
                  1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1,
                  1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1
                  ]),  # 8
        np.array([1, 1, 1, 1, 1,
                  1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1,
                  -1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1])  # 9
        ]


hnn = hopfield(Data)
plt.imshow(hnn.W)
plt.show()
# 选择预测的号码
inMat = Data[2]
plt.imshow(inMat.reshape((6, 5)))
plt.show()
# 增加0.1的噪声
for i in range(int(30 * 0.1)):
    k = random.randint(0, 29)
    inMat[k] = -inMat[k]

plt.imshow(inMat.reshape((6, 5)))
plt.show()
# 进行联想记忆
newMat = hnn.fit(inMat)

plt.imshow(newMat.reshape((6, 5)))
plt.show()