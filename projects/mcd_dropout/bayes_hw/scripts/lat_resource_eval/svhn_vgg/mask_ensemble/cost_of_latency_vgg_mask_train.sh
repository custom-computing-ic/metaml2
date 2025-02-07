# As we just need the model to test hardware performance, number of epoch is set as 10 to reduce the training time
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --gpus 0 --save_model svhn_vgg_spt_2samples_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --opt_mode spatial --mc_samples 2 --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --gpus 0 --save_model svhn_vgg_spt_3samples_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --opt_mode spatial --mc_samples 3 --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --gpus 0 --save_model svhn_vgg_spt_5samples_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --opt_mode spatial --mc_samples 5 --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --gpus 0 --save_model svhn_vgg_spt_7samples_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --opt_mode spatial --mc_samples 7 --dropout_type mask
python3 train_qkeras_mcme.py --dataset svhn --num_epoch 5 --batch_size 128 --lr 0.001 --gpus 0 --save_model svhn_vgg_spt_9samples_mask --quant_tbit 8 --model_name vgg --save_dir ./exp_svhn_bayes_vgg --opt_mode spatial --mc_samples 9 --dropout_type mask