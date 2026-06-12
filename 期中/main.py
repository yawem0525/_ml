import numpy as np
from network import NeuralNetwork
from layers import DenseLayer
# ... 匯入其他模組

# XOR 的輸入與正確答案
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

# 建立你的引擎
engine = NeuralNetwork()
engine.add(DenseLayer(2, 3)) # 輸入層2個神經元，隱藏層3個
# engine.add(ActivationLayer(sigmoid, sigmoid_derivative))
engine.add(DenseLayer(3, 1)) # 隱藏層3個，輸出層1個 (預測 0 或 1)
# engine.add(ActivationLayer(sigmoid, sigmoid_derivative))

# 開始訓練
# engine.train(X, Y, epochs=1000, learning_rate=0.1)
