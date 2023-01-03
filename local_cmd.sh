# MY_CMD="python main.py --config cifar10_sub0.yml --doc cifar10_sub0 --ni --poison --poison_path /egr/research-dselab/renjie3/renjie/diffusion/ncsnv2/exp/logs/cifar10_sub0_13/checkpoint_150000_17_adv_perturb.npy"

# MY_CMD="python main.py --sample --config cifar10_sub0.yml -i cifar10_sub0 --doc cifar10_sub0 --ni"

MY_CMD="python main.py --adv --config cifar10_sub0.yml --doc cifar10_sub0_19 --model_group --ckpt_id_gourp 100 200 --ni --adv_loss_type bilevel_gradient_matching"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='0' $MY_CMD
