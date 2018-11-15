jbsub -cores 1+1 -require k80 -mem 50g -err ../err/$1.txt -out ../out/$1.txt -q x86_6h /u/stamilse/miniconda3/bin/python3.6 train.py --scenario $2 --num-agents $3 --num-adversaries $4 --lr_actor $5 --lr_critic $6 --lr_lamda $7 --constrained True --constraint_type CVAR --cvar_alpha_adv_agent $8 --cvar_alpha_good_agent $9 --cvar_beta $10 --independent-learner $11 --exp-name $2_$3_$4_uest_False_indep_$11_constr_True_cvar_$1