from huggingface_hub import snapshot_download

# 设置下载路径
# local_dir1 = "/mnt/gemlab_data_3/User_database/liangzhichao/task2"

# # 开始下载
# snapshot_download(
#     repo_id="FLARE-MedFM/FLARE-Task2-LaptopSeg",
#     repo_type="dataset",
#     local_dir=local_dir1,
#     local_dir_use_symlinks=False,
#     resume_download=False,
# )

local_dir2 = "/mnt/gemlab_data_3/User_database/liangzhichao/task1_2D"
snapshot_download(
    repo_id="FLARE-MedFM/FLARE-Task1-Pancancer",
    repo_type="dataset",
    local_dir=local_dir2,
    local_dir_use_symlinks=False,
    resume_download=False,
)



# local_dir3 = "/mnt/gemlab_data_3/User_database/liangzhichao/task1_3D"
# snapshot_download(
#     repo_id="FLARE-MedFM/FLARE-Task1-PancancerRECIST-to-3D",
#     repo_type="dataset",
#     local_dir=local_dir3,
#     local_dir_use_symlinks=False,
#     resume_download=True,
# )
