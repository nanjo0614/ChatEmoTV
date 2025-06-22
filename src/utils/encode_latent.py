# zP/zB をダミー化したバージョン

import numpy as np
def get_latents(abc_str: str):
    z_p = np.random.randn(128).astype("float32")
    # 8 小節固定でまず作る
    z_b = np.random.randn(8, 128).astype("float32")
    return z_p, z_b
