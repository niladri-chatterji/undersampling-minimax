#!/bin/bash
# exclude certain machines and load conda environment
mynlprun="nlprun3 -x jagupard[10-20] -a is"

maj_base_samples=2500
min_base_samples=500



experiment=cifar_early_stopped_scaling_dro


for seed in 0 1 2 3 4
do
    group="cifar_scaling_minority_dro"
    for min_samples in 500 1000 1500 2000 
    do
        CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
        eval ${CMD}
        sleep 1

    done

    group="cifar_scaling_majority_dro"
    for maj_samples in 3000 3500 4000 4500
    do
        CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_samples},${min_base_samples}] logger.wandb.group=${group} seed=${seed}"\"
        eval ${CMD}
        sleep 1

    done

    group="cifar_scaling_n_dro"
    imb_factor=5
    for add_samples in 100 200 300 400 500
    do

        maj_add=$(($add_samples*$imb_factor))
        maj_samples=$(($maj_add + $maj_base_samples))
        min_samples=$(($add_samples + $min_base_samples))

        CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
        eval ${CMD}
        sleep 1

    done
done




# group="cifar_scaling_minority_dro"

# seed=0

# for min_samples in 1000
# do 
#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1
# done


# group="test"

# seed=0

# for min_samples in 1000
# do 
#     CMD="${mynlprun} \"python run.py +experiment=celeba_dro loss_fn=cross_entropy optimizer.lr=0.001 trainer.max_epochs=200 model.adv_probs_lr=0.05"\"
#     eval ${CMD}
#     sleep 1
# done




# experiment=cifar_early_stopped_scaling_tiltedloss


# group="cifar_scaling_minority_tiltedloss"

# seed=1

# for min_samples in 1000
# do 
#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=800 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1
# done

# seed=3

# for min_samples in 1000 2000
# do 
#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=800 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1
# done

# seed=4

# for min_samples in 1000 1500 2000
# do 
#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=800 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1
# done

# group="cifar_scaling_majority_tiltedloss"


# seed=0

# for maj_samples in 3500 4500
# do
#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=800 datamodule.class_samples=[${maj_samples},${min_base_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1
# done


# group="cifar_scaling_n_tiltedloss"

# seed=3

# imb_factor=5
# for add_samples in 100
# do

#     maj_add=$(($add_samples*$imb_factor))
#     maj_samples=$(($maj_add + $maj_base_samples))
#     min_samples=$(($add_samples + $min_base_samples))

#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=800 datamodule.class_samples=[${maj_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1

# done




# for seed in 0 1 2 3 4
# do
#     group="cifar_scaling_minority_tiltedloss"
#     for min_samples in 500 1000 1500 2000 
#     do
#         CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#         eval ${CMD}
#         sleep 1

#     done

#     group="cifar_scaling_majority_tiltedloss"
#     for maj_samples in 3000 3500 4000 4500
#     do
#         CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_samples},${min_base_samples}] logger.wandb.group=${group} seed=${seed}"\"
#         eval ${CMD}
#         sleep 1

#     done

#     group="cifar_scaling_n_tiltedloss"
#     imb_factor=5
#     for add_samples in 100 200 300 400 500
#     do

#         maj_add=$(($add_samples*$imb_factor))
#         maj_samples=$(($maj_add + $maj_base_samples))
#         min_samples=$(($add_samples + $min_base_samples))

#         CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#         eval ${CMD}
#         sleep 1

#     done
# done

### seed 2
# seed=2

# # # first set of experiments scaling minority group samples with vsloss
# group="cifar_scaling_minority_tiltedloss"

# for min_samples in 1500
# do
#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1

# done









# experiment=cifar_reweighted_early_stopped_scaling_vsloss







# ### seed 2
# seed=2

# # # first set of experiments scaling minority group samples with vsloss
# group="cifar_scaling_minority_vsloss"

# for min_samples in 1500
# do
#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1

# done

# # # second set of experiments scaling minority group samples with vsloss
# group="cifar_scaling_majority_vsloss"

# for maj_samples in 3000 3500 4000
# do
#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_samples},${min_base_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1

# done


### seed 3

# seed=3

# # third set of experiments scaling majority and minority group samples with vsloss
# group="cifar_scaling_n_vsloss"
# imb_factor=5
# for add_samples in 300 400
# do

#     maj_add=$(($add_samples*$imb_factor))
#     maj_samples=$(($maj_add + $maj_base_samples))
#     min_samples=$(($add_samples + $min_base_samples))

#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1

# done


### seed 4

# seed=4

# # # first set of experiments scaling minority group samples with vsloss
# group="cifar_scaling_minority_vsloss"

# for min_samples in 1500
# do
#     CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#     eval ${CMD}
#     sleep 1

# done

# for seed in 1 2 3 4
# do

#     experiment=cifar_reweighted_early_stopped_scaling


#     # # first set of experiments scaling minority group samples with CE+IW+ES
#     group="cifar_scaling_minority_ermrw"

#     for min_samples in 500 1000 1500 2000 
#     do
#         CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#         eval ${CMD}
#         sleep 1

#     done

#     # # second set of experiments scaling minority group samples with CE+IW+ES
#     group="cifar_scaling_majority_ermrw"

#     for maj_samples in 3000 3500 4000 4500
#     do
#         CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_samples},${min_base_samples}] logger.wandb.group=${group} seed=${seed}"\"
#         eval ${CMD}
#         sleep 1

#     done


#     # third set of experiments scaling majority and minority group samples with CE+IW+ES
#     group="cifar_scaling_n_ermrw"
#     imb_factor=5
#     for add_samples in 100 200 300 400 500
#     do

#         maj_add=$(($add_samples*$imb_factor))
#         maj_samples=$(($maj_add + $maj_base_samples))
#         min_samples=$(($add_samples + $min_base_samples))

#         CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#         eval ${CMD}
#         sleep 1

#     done
    
#     experiment=cifar_reweighted_early_stopped_scaling_vsloss


#     # # first set of experiments scaling minority group samples with vsloss
#     group="cifar_scaling_minority_vsloss"

#     for min_samples in 500 1000 1500 2000 
#     do
#         CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_base_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#         eval ${CMD}
#         sleep 1

#     done

#     # # second set of experiments scaling minority group samples with vsloss
#     group="cifar_scaling_majority_vsloss"

#     for maj_samples in 3000 3500 4000 4500
#     do
#         CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_samples},${min_base_samples}] logger.wandb.group=${group} seed=${seed}"\"
#         eval ${CMD}
#         sleep 1

#     done


#     # third set of experiments scaling majority and minority group samples with vsloss
#     group="cifar_scaling_n_vsloss"
#     imb_factor=5
#     for add_samples in 100 200 300 400 500
#     do

#         maj_add=$(($add_samples*$imb_factor))
#         maj_samples=$(($maj_add + $maj_base_samples))
#         min_samples=$(($add_samples + $min_base_samples))

#         CMD="${mynlprun} \"python run.py +experiment=${experiment} trainer.max_epochs=400 datamodule.class_samples=[${maj_samples},${min_samples}] logger.wandb.group=${group} seed=${seed}"\"
#         eval ${CMD}
#         sleep 1

#     done
    
    
#done



# for seed in 1
# do

#     CMD="${mynlprun} \"python run.py +experiment=cifar_undersampled loss_fn=cross_entropy trainer.max_epochs=400 seed=${seed}"\"
#     eval ${CMD}
#     sleep 1

#     CMD="${mynlprun} \"python run.py +experiment=cifar_reweighted_early_stopped loss_fn=cross_entropy trainer.max_epochs=400 seed=${seed}"\"
#     eval ${CMD}
#     sleep 1

# done
# CMD="${mynlprun} \"python run.py +experiment=celeba_undersampled loss_fn=cross_entropy trainer.max_epochs=400"\"
# eval ${CMD}
# sleep 1

# CMD="${mynlprun} \"python run.py +experiment=celeba_reweighted_early_stopped loss_fn=cross_entropy trainer.max_epochs=400"\"
# eval ${CMD}
# sleep 1

# CMD="${mynlprun} \"python run.py +experiment=celeba_erm loss_fn=cross_entropy"\"
# eval ${CMD}
# sleep 1