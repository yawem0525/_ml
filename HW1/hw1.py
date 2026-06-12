import random

def get_height(path, dist_matrix):
    """
    計算路徑的 height = -(總距離)
    """
    total_distance = 0
    n = len(path)
    for i in range(n):
        # path[i-1] 到 path[i] 的距離 (當 i=0 時，path[-1] 就是最後一個點，完美形成迴路)
        from_node = path[i - 1]
        to_node = path[i]
        total_distance += dist_matrix[from_node][to_node]
    
    return -total_distance

def get_neighbor(path):
    """
    產生鄰居：使用 2-opt 交換法
    """
    new_path = path.copy()
    n = len(new_path)
    
    # 隨機挑選兩個不相鄰的切斷點
    i, j = sorted(random.sample(range(n), 2))
    
    # 將 i 到 j 之間的路徑反轉，這在幾何上等同於打斷兩條邊並重新交叉連接
    new_path[i:j] = reversed(new_path[i:j])
    
    return new_path

def generate_initial_solution(n):
    """
    初始解 1 => 2 => 3 => ... => n
    """
    return list(range(n))

# --- 測試範例 ---
# 假設有 5 個城市 (n=5)
n = 5
current_solution = generate_initial_solution(n)
print(f"初始解: {current_solution}")

# 產生一個鄰居看看
neighbor_solution = get_neighbor(current_solution)
print(f"鄰居解: {neighbor_solution}")
