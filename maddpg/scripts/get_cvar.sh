rm -f scratch.sh
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 0\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 0"
grep 'cvar: [\-]*[0-9\.]*,' -o scratch.txt | cut -f2 -d' ' | awk 'BEGIN{avg=0.0} {avg+=$0} END{ print avg/NR}'
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 1\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 1"
grep 'cvar: [\-]*[0-9\.]*,' -o scratch.txt | cut -f2 -d' ' | awk 'BEGIN{avg=0.0} {avg+=$0} END{ print avg/NR}'
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 2\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 2"
grep 'cvar: [\-]*[0-9\.]*,' -o scratch.txt | cut -f2 -d' ' | awk 'BEGIN{avg=0.0} {avg+=$0} END{ print avg/NR}'
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 3\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 3"
grep 'cvar: [\-]*[0-9\.]*,' -o scratch.txt | cut -f2 -d' ' | awk 'BEGIN{avg=0.0} {avg+=$0} END{ print avg/NR}'
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 4\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 4"
grep 'cvar: [\-]*[0-9\.]*,' -o scratch.txt | cut -f2 -d' ' | awk 'BEGIN{avg=0.0} {avg+=$0} END{ print avg/NR}'
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 5\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 5"
grep 'cvar: [\-]*[0-9\.]*,' -o scratch.txt | cut -f2 -d' ' | awk 'BEGIN{avg=0.0} {avg+=$0} END{ print avg/NR}'
rm -f scratch.sh 
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 6\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 6"
grep 'cvar: [\-]*[0-9\.]*,' -o scratch.txt | cut -f2 -d' ' | awk 'BEGIN{avg=0.0} {avg+=$0} END{ print avg/NR}'
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 7\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 7"
grep 'cvar: [\-]*[0-9\.]*,' -o scratch.txt | cut -f2 -d' ' | awk 'BEGIN{avg=0.0} {avg+=$0} END{ print avg/NR}'
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 8\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 8"
grep 'cvar: [\-]*[0-9\.]*,' -o scratch.txt | cut -f2 -d' ' | awk 'BEGIN{avg=0.0} {avg+=$0} END{ print avg/NR}'
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 9\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 9"
grep 'cvar: [\-]*[0-9\.]*,' -o scratch.txt | cut -f2 -d' ' | awk 'BEGIN{avg=0.0} {avg+=$0} END{ print avg/NR}'
rm -f scratch.sh
rm -f scratch.txt
