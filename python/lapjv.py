import lap
import numpy as np
import time


if __name__ == "__main__":
    array = np.random.rand(4, 5)
    start = time.time()
    result = lap.lapjv(array, extend_cost=True)
    end = time.time()
    print(f"lapjv: Time={end - start:.8f}s")
    print(result)
