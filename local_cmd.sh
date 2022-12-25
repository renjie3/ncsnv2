# MY_CMD="python main.py --config cifar10.yml --doc cifar10"

MY_CMD="python main.py --sample --config cifar10.yml -i cifar10 --doc test_cifar10"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='0' $MY_CMD
