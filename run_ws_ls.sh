
ls_epochs=0
start_seed=10
end_seed=$(($start_seed+20))

for seed in $(seq $start_seed $end_seed)
do
    save_dir=/home/ubuntu/nasbench-1shot1_crwhite/experiments/ft_$ls_epochs\_$seed
    echo $save_dir
    #python optimizers/random_search_with_weight_sharing/random_weight_share.py --save_dir $save_dir --seed $seed --ls_epochs $ls_epochs
done
