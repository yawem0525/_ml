import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        # 隨機初始化權重矩陣 W 和偏差 b
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
        # 用來暫存前向傳播的輸入與輸出
        self.input = None
        
    def forward(self, input_data):
        self.input = input_data
        # 實作矩陣內積： Z = X * W + b
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        # 這裡將實作最困難的反向傳播邏輯
        # 1. 計算當前層的權重梯度
        # 2. 更新 self.weights 和 self.bias
        # 3. 計算並回傳傳遞給上一層的輸入梯度
        pass
