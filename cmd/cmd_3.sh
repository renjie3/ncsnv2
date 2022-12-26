cd /egr/research-dselab/renjie3/renjie/diffusion/ncsnv2
MY_CMD="python main.py --config cifar10_sub0.yml --doc cifar10_sub0 --ni --job_id 3 "
CUDA_VISIBLE_DEVICES='1' ${MY_CMD}

if [ $? -eq 0 ];then
echo -e "grandriver JobID:3 \n Python_command: \n python main.py --config cifar10_sub0.yml --doc cifar10_sub0 --ni \n " | mail -s "[Done] grandriver " renjie2179@outlook.com
else
echo -e "grandriver JobID:3 \n Python_command: \n python main.py --config cifar10_sub0.yml --doc cifar10_sub0 --ni \n " | mail -s "[Fail] grandriver " renjie2179@outlook.com
fi
