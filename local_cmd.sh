# MY_CMD="python main.py --config cifar10_sub0.yml --doc cifar10_sub0 --ni"

# MY_CMD="python main.py --sample --config cifar10_sub0.yml -i cifar10_sub0 --doc cifar10_sub0 --ni"

MY_CMD="python main.py --adv --config cifar10_sub0.yml --doc cifar10_sub0_13 --ni --adv_loss_type bilevel_min_forward_loss"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='0, 2' $MY_CMD
