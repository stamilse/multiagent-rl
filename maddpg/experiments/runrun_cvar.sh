####jbsub -cores 1+1 -require k80 -mem 50g -err ../err/$2_$3_$4_uest_${11}_indep_${10}_constr_True_expvar_$1.txt -out ../out/$2_$3_$4_uest_${11}_indep_${10}_constr_True_expvar_$1.txt -q x86_6h /u/stamilse/miniconda3/bin/python3.6 train.py --scenario $2 --num-agents $3 --num-adversaries $4 --lr_actor $5 --lr_critic $6 --lr_lamda $7 --constrained True --constraint_type CVAR --cvar_alpha_adv_agent $8 --cvar_alpha_good_agent $9 --independent-learner ${10} --u_estimation ${11} --exp-name $2_$3_$4_uest_${11}_indep_${10}_constr_True_cvar_$1

./run_cvar.sh $1 $2 2 1 0.005 0.01 0.0001 4.7373 -0.2491305 False False
./run_cvar.sh $1 $2 3 1 0.005 0.01 0.0001 1.048756 -0.339814 False False
./run_cvar.sh $1 $2 3 2 0.005 0.01 0.0001 5.71536 5.71536 False False
#./run_cvar.sh $1 $2 4 2 0.005 0.01 0.0001 False False
#./run_cvar.sh $1 $2 7 2 0.005 0.01 0.0001 0.117641 False False
#./run_cvar.sh $1 $2 7 3 0.005 0.01 0.0001 0.105336 False False
#./run_cvar.sh $1 $2 7 4 0.005 0.01 0.0001 0.066746 False False
#./run_cvar.sh $1 $2 10 5 0.005 0.01 0.0001 0.082459 False False

./run_cvar.sh $1 $2 2 1 0.005 0.01 0.0001 4.86032 -0.20103 False True
./run_cvar.sh $1 $2 3 1 0.005 0.01 0.0001 1.14375 -0.17 False True
./run_cvar.sh $1 $2 3 2 0.005 0.01 0.0001 5.4518 -0.1333235 False True
#./run_cvar.sh $1 $2 4 2 0.005 0.01 0.0001 0.60368 False True
#./run_cvar.sh $1 $2 7 2 0.005 0.01 0.0001 1.4936 False True
#./run_cvar.sh $1 $2 7 3 0.005 0.01 0.0001 1.239505 False True
#./run_cvar.sh $1 $2 7 4 0.005 0.01 0.0001 1.350625 False True

