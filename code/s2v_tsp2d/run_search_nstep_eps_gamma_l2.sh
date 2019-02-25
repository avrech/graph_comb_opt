#!/bin/bash
l2_lambdas=(
    0.00 #original
    0.0001
)
gammas=(
    1
    0.9
    0.1 #original
)
epsilons_start=(
    0.9
    1 #original
)
epsilons_end=(
    0.01
    0.1
    1 #original
)
n_steps=(
    10
    5
    1 #original
)
for l2 in "${l2_lambdas[@]}"; do
    for decay in "${gammas[@]}"; do
        for epsilon_greedy_start in "${epsilons_start[@]}"; do
            for epsilon_greedy_end in "${epsilons_end[@]}"; do
                for n_step in "${n_steps[@]}"; do
                    #decay=0.99 # gamma - discount factor
                    #epsilon_greedy_start=1
                    #epsilon_greedy_end=0.01
                    # nstep
                    #n_step=1

                    g_type=clustered

                    # max belief propagation iteration
                    max_bp_iter=4

                    # embedding size
                    embed_dim=64

                    # gpu card id
                    dev_id=1

                    # max batch size for training/testing
                    batch_size=128

                    net_type=QNet
                    # set reg_hidden=0 to make a linear regression
                    reg_hidden=32

                    # learning rate
                    learning_rate=0.0001

                    # init weights with rand normal(0, w_scale)
                    w_scale=0.01


                    knn=10

                    min_n=40
                    max_n=50

                    num_env=1
                    mem_size=50000

                    max_iter=200000

                    # folder to save the trained model
                    result_root=results/dqn-$g_type-$min_n-$max_n-nstep-$n_step-gamma-$decay-l2-$l2-epsilon-$epsilon_greedy_start-to-$epsilon_greedy_end

                    save_dir=$result_root/ntype-$net_type-embed-$embed_dim-nbp-$max_bp_iter-rh-$reg_hidden
                    echo "model save dir: $save_dir"

                    if [ ! -e $save_dir ];
                    then
                        mkdir -p $save_dir
                    fi

                    python main.py \
                        -net_type $net_type \
                        -n_step $n_step \
                        -data_root ../../data/tsp2d \
                        -decay $decay \
                        -epsilon_greedy_start $epsilon_greedy_start \
                        -epsilon_greedy_end $epsilon_greedy_end \
                        -knn $knn \
                        -min_n $min_n \
                        -max_n $max_n \
                        -num_env $num_env \
                        -max_iter $max_iter \
                        -mem_size $mem_size \
                        -g_type $g_type \
                        -learning_rate $learning_rate \
                        -max_bp_iter $max_bp_iter \
                        -net_type $net_type \
                        -max_iter $max_iter \
                        -save_dir $save_dir \
                        -embed_dim $embed_dim \
                        -batch_size $batch_size \
                        -reg_hidden $reg_hidden \
                        -momentum 0.9 \
                        -l2 $l2 \
                        -w_scale $w_scale \
                        2>&1 | tee $save_dir/log-$min_n-${max_n}.txt

                done
            done
        done
    done
done
