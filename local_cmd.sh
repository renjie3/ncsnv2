# MY_CMD="python main.py --config cifar10_sub0.yml --doc cifar10_sub0 --ni"

# MY_CMD="python main.py --sample --config cifar10.yml -i cifar10 --doc test_cifar10"

MY_CMD="python main.py --adv --config cifar10_sub0.yml --doc test_cifar10 --ni"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='0' $MY_CMD
