./get_variance.sh $1 > x
awk 'BEGIN{avg=0.0; count=0} { if($1!="For"){avg+=$0; count+=1}} END{print avg/count}' x
rm x
