jbsub -cores 1+1 -require k80 -mem 50g -err ../err/$1.txt -out ../out/$1.txt -q x86_6h /u/stamilse/miniconda3/bin/python3.6 train.py --scenario $2 --num-a
gents $3 --num-adversaries $4 --lr_actor $5 --lr_critic $6 --lr_lamda $7 --constrained True --constraint_type Exp_Var --exp_var_alpha $8 --independent-lea
rner $9 --exp-name $1


