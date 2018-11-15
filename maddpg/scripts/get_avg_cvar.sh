./get_cvar.sh $1 > x

grep -v 'For' x > x1
head -$2 x1 > x_adversary
num_adv=`expr $2 + 1`
tail -n +$num_adv x1 > x_good
echo 'For adversary'
awk 'BEGIN{avg=0.0} { avg+=$0} END{print avg/NR}' x_adversary
echo 'For good'
awk 'BEGIN{avg=0.0} { avg+=$0} END{print avg/NR}' x_good
rm x1
