######jbsub -cores 1+1 -require k80 -mem 80g -err ../err/$2_$3_$4_uest_$9_indep_$8_constr_False_none_$1.txt -out ../out/$2_$3_$4_uest_$9_indep_$8_constr_False_none_$1.txt -q x86_12h /u/stamilse/miniconda3/bin/python3.6 train.py --scenario $2 --num-agents $3 --num-adversaries $4 --lr_actor $5 --lr_critic $6 --lr_lamda $7 --independent-learner $8 --constrained False --u_estimation $9 --exp-name $2_$3_$4_uest_$9_indep_$8_constr_False_none_$1

./run_unconstrained.sh $1 $2 2 1 0.005 0.01 0.0001 False False
./run_unconstrained.sh $1 $2 3 1 0.005 0.01 0.0001 False False
./run_unconstrained.sh $1 $2 3 2 0.005 0.01 0.0001 False False
./run_unconstrained.sh $1 $2 4 2 0.005 0.01 0.0001 False False
./run_unconstrained.sh $1 $2 7 2 0.005 0.01 0.0001 False False
./run_unconstrained.sh $1 $2 7 3 0.005 0.01 0.0001 False False
./run_unconstrained.sh $1 $2 7 4 0.005 0.01 0.0001 False False
./run_unconstrained.sh $1 $2 10 5 0.005 0.01 0.0001 False False

./run_unconstrained.sh $1 $2 2 1 0.005 0.01 0.0001 False True
./run_unconstrained.sh $1 $2 3 1 0.005 0.01 0.0001 False True
./run_unconstrained.sh $1 $2 3 2 0.005 0.01 0.0001 False True
./run_unconstrained.sh $1 $2 4 2 0.005 0.01 0.0001 False True
./run_unconstrained.sh $1 $2 7 2 0.005 0.01 0.0001 False True
./run_unconstrained.sh $1 $2 7 3 0.005 0.01 0.0001 False True
./run_unconstrained.sh $1 $2 7 4 0.005 0.01 0.0001 False True
./run_unconstrained.sh $1 $2 10 5 0.005 0.01 0.0001 False True


./run_unconstrained.sh $1 $2 2 1 0.005 0.01 0.0001 True True
./run_unconstrained.sh $1 $2 3 1 0.005 0.01 0.0001 True True
./run_unconstrained.sh $1 $2 3 2 0.005 0.01 0.0001 True True
./run_unconstrained.sh $1 $2 4 2 0.005 0.01 0.0001 True True
./run_unconstrained.sh $1 $2 7 2 0.005 0.01 0.0001 True True
./run_unconstrained.sh $1 $2 7 3 0.005 0.01 0.0001 True True
./run_unconstrained.sh $1 $2 7 4 0.005 0.01 0.0001 True True
./run_unconstrained.sh $1 $2 10 5 0.005 0.01 0.0001 True True
