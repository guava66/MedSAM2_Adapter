import os
import sys
from torch.distributed.run import main as torchrun_main

if __name__ == "__main__":
    # 修改 sys.argv 来模拟 torchrun 的参数
    sys.argv = [
                   "torchrun",
                   "--nproc_per_node=2",  # 设置每节点的进程数量
                   "--nnodes=1",  # 设置总节点数
                   "--node_rank=0",  # 当前节点编号
                   "/data/pyhData/MedSAM-MedSAM2/finetune_sam2_img.py",  # 你的训练脚本路径
               ] + sys.argv[1:]  # 保留其他可能的参数

    # 调用 torchrun 的主入口
    torchrun_main()
