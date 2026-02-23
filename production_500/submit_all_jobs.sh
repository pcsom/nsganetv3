#!/bin/bash

cd /home/hice1/glu49/nsganetv3

echo "Submitting 500 training jobs..."

job_ids=()

job_id=$(sbatch production_500/arch_0000/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0000/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0001/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0001/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0002/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0002/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0003/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0003/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0004/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0004/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0005/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0005/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0006/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0006/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0007/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0007/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0008/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0008/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0009/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0009/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0010/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0010/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0011/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0011/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0012/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0012/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0013/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0013/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0014/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0014/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0015/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0015/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0016/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0016/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0017/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0017/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0018/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0018/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0019/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0019/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0020/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0020/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0021/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0021/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0022/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0022/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0023/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0023/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0024/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0024/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0025/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0025/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0026/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0026/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0027/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0027/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0028/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0028/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0029/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0029/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0030/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0030/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0031/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0031/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0032/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0032/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0033/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0033/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0034/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0034/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0035/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0035/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0036/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0036/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0037/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0037/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0038/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0038/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0039/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0039/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0040/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0040/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0041/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0041/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0042/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0042/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0043/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0043/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0044/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0044/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0045/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0045/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0046/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0046/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0047/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0047/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0048/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0048/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0049/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0049/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0050/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0050/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0051/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0051/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0052/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0052/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0053/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0053/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0054/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0054/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0055/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0055/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0056/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0056/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0057/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0057/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0058/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0058/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0059/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0059/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0060/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0060/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0061/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0061/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0062/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0062/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0063/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0063/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0064/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0064/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0065/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0065/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0066/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0066/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0067/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0067/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0068/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0068/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0069/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0069/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0070/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0070/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0071/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0071/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0072/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0072/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0073/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0073/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0074/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0074/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0075/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0075/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0076/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0076/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0077/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0077/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0078/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0078/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0079/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0079/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0080/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0080/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0081/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0081/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0082/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0082/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0083/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0083/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0084/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0084/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0085/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0085/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0086/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0086/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0087/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0087/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0088/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0088/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0089/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0089/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0090/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0090/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0091/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0091/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0092/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0092/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0093/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0093/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0094/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0094/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0095/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0095/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0096/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0096/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0097/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0097/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0098/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0098/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0099/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0099/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0100/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0100/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0101/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0101/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0102/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0102/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0103/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0103/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0104/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0104/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0105/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0105/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0106/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0106/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0107/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0107/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0108/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0108/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0109/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0109/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0110/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0110/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0111/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0111/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0112/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0112/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0113/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0113/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0114/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0114/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0115/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0115/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0116/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0116/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0117/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0117/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0118/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0118/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0119/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0119/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0120/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0120/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0121/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0121/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0122/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0122/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0123/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0123/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0124/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0124/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0125/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0125/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0126/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0126/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0127/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0127/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0128/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0128/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0129/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0129/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0130/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0130/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0131/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0131/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0132/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0132/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0133/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0133/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0134/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0134/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0135/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0135/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0136/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0136/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0137/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0137/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0138/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0138/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0139/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0139/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0140/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0140/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0141/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0141/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0142/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0142/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0143/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0143/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0144/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0144/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0145/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0145/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0146/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0146/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0147/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0147/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0148/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0148/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0149/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0149/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0150/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0150/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0151/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0151/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0152/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0152/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0153/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0153/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0154/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0154/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0155/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0155/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0156/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0156/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0157/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0157/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0158/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0158/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0159/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0159/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0160/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0160/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0161/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0161/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0162/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0162/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0163/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0163/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0164/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0164/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0165/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0165/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0166/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0166/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0167/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0167/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0168/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0168/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0169/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0169/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0170/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0170/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0171/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0171/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0172/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0172/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0173/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0173/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0174/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0174/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0175/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0175/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0176/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0176/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0177/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0177/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0178/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0178/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0179/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0179/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0180/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0180/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0181/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0181/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0182/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0182/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0183/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0183/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0184/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0184/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0185/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0185/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0186/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0186/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0187/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0187/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0188/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0188/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0189/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0189/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0190/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0190/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0191/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0191/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0192/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0192/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0193/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0193/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0194/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0194/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0195/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0195/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0196/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0196/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0197/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0197/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0198/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0198/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0199/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0199/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0200/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0200/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0201/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0201/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0202/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0202/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0203/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0203/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0204/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0204/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0205/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0205/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0206/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0206/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0207/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0207/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0208/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0208/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0209/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0209/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0210/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0210/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0211/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0211/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0212/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0212/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0213/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0213/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0214/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0214/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0215/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0215/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0216/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0216/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0217/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0217/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0218/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0218/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0219/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0219/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0220/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0220/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0221/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0221/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0222/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0222/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0223/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0223/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0224/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0224/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0225/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0225/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0226/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0226/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0227/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0227/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0228/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0228/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0229/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0229/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0230/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0230/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0231/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0231/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0232/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0232/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0233/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0233/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0234/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0234/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0235/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0235/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0236/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0236/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0237/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0237/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0238/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0238/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0239/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0239/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0240/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0240/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0241/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0241/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0242/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0242/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0243/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0243/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0244/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0244/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0245/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0245/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0246/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0246/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0247/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0247/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0248/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0248/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0249/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0249/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0250/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0250/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0251/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0251/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0252/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0252/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0253/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0253/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0254/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0254/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0255/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0255/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0256/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0256/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0257/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0257/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0258/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0258/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0259/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0259/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0260/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0260/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0261/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0261/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0262/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0262/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0263/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0263/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0264/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0264/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0265/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0265/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0266/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0266/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0267/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0267/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0268/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0268/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0269/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0269/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0270/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0270/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0271/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0271/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0272/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0272/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0273/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0273/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0274/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0274/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0275/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0275/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0276/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0276/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0277/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0277/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0278/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0278/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0279/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0279/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0280/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0280/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0281/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0281/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0282/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0282/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0283/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0283/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0284/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0284/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0285/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0285/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0286/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0286/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0287/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0287/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0288/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0288/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0289/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0289/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0290/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0290/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0291/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0291/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0292/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0292/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0293/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0293/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0294/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0294/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0295/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0295/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0296/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0296/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0297/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0297/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0298/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0298/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0299/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0299/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0300/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0300/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0301/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0301/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0302/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0302/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0303/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0303/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0304/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0304/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0305/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0305/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0306/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0306/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0307/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0307/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0308/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0308/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0309/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0309/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0310/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0310/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0311/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0311/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0312/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0312/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0313/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0313/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0314/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0314/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0315/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0315/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0316/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0316/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0317/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0317/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0318/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0318/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0319/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0319/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0320/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0320/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0321/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0321/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0322/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0322/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0323/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0323/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0324/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0324/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0325/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0325/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0326/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0326/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0327/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0327/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0328/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0328/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0329/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0329/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0330/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0330/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0331/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0331/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0332/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0332/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0333/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0333/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0334/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0334/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0335/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0335/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0336/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0336/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0337/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0337/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0338/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0338/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0339/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0339/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0340/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0340/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0341/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0341/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0342/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0342/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0343/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0343/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0344/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0344/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0345/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0345/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0346/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0346/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0347/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0347/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0348/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0348/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0349/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0349/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0350/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0350/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0351/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0351/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0352/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0352/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0353/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0353/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0354/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0354/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0355/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0355/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0356/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0356/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0357/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0357/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0358/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0358/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0359/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0359/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0360/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0360/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0361/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0361/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0362/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0362/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0363/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0363/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0364/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0364/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0365/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0365/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0366/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0366/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0367/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0367/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0368/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0368/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0369/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0369/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0370/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0370/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0371/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0371/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0372/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0372/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0373/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0373/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0374/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0374/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0375/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0375/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0376/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0376/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0377/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0377/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0378/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0378/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0379/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0379/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0380/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0380/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0381/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0381/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0382/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0382/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0383/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0383/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0384/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0384/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0385/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0385/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0386/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0386/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0387/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0387/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0388/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0388/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0389/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0389/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0390/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0390/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0391/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0391/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0392/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0392/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0393/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0393/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0394/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0394/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0395/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0395/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0396/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0396/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0397/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0397/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0398/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0398/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0399/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0399/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0400/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0400/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0401/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0401/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0402/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0402/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0403/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0403/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0404/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0404/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0405/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0405/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0406/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0406/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0407/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0407/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0408/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0408/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0409/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0409/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0410/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0410/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0411/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0411/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0412/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0412/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0413/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0413/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0414/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0414/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0415/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0415/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0416/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0416/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0417/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0417/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0418/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0418/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0419/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0419/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0420/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0420/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0421/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0421/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0422/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0422/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0423/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0423/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0424/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0424/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0425/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0425/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0426/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0426/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0427/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0427/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0428/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0428/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0429/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0429/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0430/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0430/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0431/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0431/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0432/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0432/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0433/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0433/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0434/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0434/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0435/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0435/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0436/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0436/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0437/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0437/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0438/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0438/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0439/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0439/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0440/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0440/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0441/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0441/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0442/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0442/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0443/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0443/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0444/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0444/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0445/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0445/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0446/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0446/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0447/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0447/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0448/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0448/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0449/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0449/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0450/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0450/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0451/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0451/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0452/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0452/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0453/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0453/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0454/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0454/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0455/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0455/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0456/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0456/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0457/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0457/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0458/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0458/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0459/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0459/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0460/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0460/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0461/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0461/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0462/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0462/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0463/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0463/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0464/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0464/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0465/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0465/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0466/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0466/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0467/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0467/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0468/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0468/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0469/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0469/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0470/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0470/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0471/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0471/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0472/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0472/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0473/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0473/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0474/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0474/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0475/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0475/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0476/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0476/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0477/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0477/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0478/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0478/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0479/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0479/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0480/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0480/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0481/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0481/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0482/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0482/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0483/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0483/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0484/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0484/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0485/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0485/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0486/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0486/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0487/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0487/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0488/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0488/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0489/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0489/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0490/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0490/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0491/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0491/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0492/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0492/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0493/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0493/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0494/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0494/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0495/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0495/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0496/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0496/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0497/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0497/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0498/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0498/train_job.sh: $job_id"
sleep 0.1

job_id=$(sbatch production_500/arch_0499/train_job.sh | awk '{print $NF}')
job_ids+=($job_id)
echo "Submitted production_500/arch_0499/train_job.sh: $job_id"
sleep 0.1


echo "All jobs submitted!"
echo "Job IDs: ${job_ids[@]}"
echo "Monitor with: squeue -u $USER"
