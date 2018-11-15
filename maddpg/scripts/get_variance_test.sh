rm -f scratch.sh
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 0\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 0"
grep 'variance: [0-9\.e\-]*,' -o scratch.txt | cut -f2 -d' ' | sort -n 
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 1\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 1"
grep 'variance: [0-9\.e\-]*,' -o scratch.txt | cut -f2 -d' ' | sort -n 
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 2\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 2"
grep 'variance: [0-9\.e\-]*,' -o scratch.txt | cut -f2 -d' ' | sort -n 
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 3\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 3"
grep 'variance: [0-9\.e\-]*,' -o scratch.txt | cut -f2 -d' ' | sort -n 
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 4\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 4"
grep 'variance: [0-9\.e\-]*,' -o scratch.txt | cut -f2 -d' ' | sort -n 
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 5\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 5"
grep 'variance: [0-9\.e\-]*,' -o scratch.txt | cut -f2 -d' ' | sort -n 
rm -f scratch.sh 
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 6\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 6"
grep 'variance: [0-9\.e\-]*,' -o scratch.txt | cut -f2 -d' ' | sort -n 
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 7\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 7"
grep 'variance: [0-9\.e\-]*,' -o scratch.txt | cut -f2 -d' ' | sort -n 
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 8\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 8"
grep 'variance: [0-9\.e\-]*,' -o scratch.txt | cut -f2 -d' ' | sort -n 
rm -f scratch.sh
rm -f scratch.txt
ls ../out/$1*.txt | awk '{ print "grep \"Running avgs for agent 9\" "$0" | tail -1"}' >> scratch.sh
chmod +x scratch.sh
./scratch.sh >> scratch.txt
echo "For agent 9"
grep 'variance: [0-9\.e\-]*,' -o scratch.txt | cut -f2 -d' ' | sort -n 
rm -f scratch.sh
rm -f scratch.txt
