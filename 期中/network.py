class NeuralNetwork:
    def __init__(self):
        self.layers = [] # 用來存放你加入的 DenseLayer 和 激勵函數

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        # 將資料依序穿過每一層 (Forward Propagation)
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, learning_rate):
        # 訓練迴圈
        for epoch in range(epochs):
            # 1. 前向傳播取得預測值
            # 2. 透過 losses.py 計算誤差
            # 3. 反向傳播：從最後一層往回推，依序呼叫 layer.backward()
            pass
