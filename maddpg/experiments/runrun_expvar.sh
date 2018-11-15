####jbsub -cores 1+1 -require k80 -mem 50g -err ../err/$1.txt -out ../out/$1.txt -q x86_6h /u/stamilse/miniconda3/bin/python3.6 train.py --scenario $2 --num-agents $3 --num-adversaries $4 --lr_actor $5 --lr_critic $6 --lr_lamda $7 --constrained True --constraint_type Exp_Var --exp_var_alpha $8 --independent-learner $9 --u_estimation $10 --exp-name $2_$3_$4_uest_$10_indep_$9_constr_True_expvar_$1

./run_expvar.sh $1 $2 2 1 0.005 0.01 0.0001 0.02810055 False False
./run_expvar.sh $1 $2 3 1 0.005 0.01 0.0001 0.056852 False False
./run_expvar.sh $1 $2 3 2 0.005 0.01 0.0001 0.03609065 False False
./run_expvar.sh $1 $2 4 2 0.005 0.01 0.0001 0.05823 False False
./run_expvar.sh $1 $2 7 2 0.005 0.01 0.0001 0.117641 False False
./run_expvar.sh $1 $2 7 3 0.005 0.01 0.0001 0.105336 False False
./run_expvar.sh $1 $2 7 4 0.005 0.01 0.0001 0.066746 False False
./run_expvar.sh $1 $2 10 5 0.005 0.01 0.0001 0.082459 False False

./run_expvar.sh $1 $2 2 1 0.005 0.01 0.0001 0.160072 False True
./run_expvar.sh $1 $2 3 1 0.005 0.01 0.0001 0.532455 False True
./run_expvar.sh $1 $2 3 2 0.005 0.01 0.0001 0.3311965 False True
./run_expvar.sh $1 $2 4 2 0.005 0.01 0.0001 0.60368 False True
./run_expvar.sh $1 $2 7 2 0.005 0.01 0.0001 1.4936 False True
./run_expvar.sh $1 $2 7 3 0.005 0.01 0.0001 1.239505 False True
./run_expvar.sh $1 $2 7 4 0.005 0.01 0.0001 1.350625 False True

