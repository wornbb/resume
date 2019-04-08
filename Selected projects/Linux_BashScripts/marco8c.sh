#!/usr/bin/zsh

cd $dump
base=$dump/${1}
base2c=${base}2c
work=${base2c}m5
echo $work
m5stat=$work/stats.txt
m5conf=$work/config.json
templ=$template/mcpat-ruby_22nm_v13_steady_1cycle.xml
#templ=$template/straight_1cycle.xml
xml=${work: : -2}xml
# 2c files, we only gem5 2c models.
op2c=${base2c}.op
pt2c=${base2c}.ptrace
# 4c files we interpolate 2c data for 4c analysis
core=8c
ptNew=${base}${core}.ptrace
gridIR=${base}${core}.gridIR
vtrace=${base}${core}.vtrace
powerG=${base}${core}.powerG
current=${base}${core}.current
# Cleanning
#rm -rf $gridIR $vtrace $powerG power_grid.test
# we assume the ptrace is already filled
#$simulation/utils/MulPtrace.py -i $pt2c -o $ptNew -n 8 
#timeout 9h $software/voltspot20/voltspot  -c $software/voltspot20/pdn4c.config -f $des/vp/chips/22nm/Penryn22_ruby_ya_8c_v13.flp -p $ptNew -v $vtrace -gridvol_file $gridIR > buffer
mv ${gridIR}powerG $powerG
python $analysis/pyscripts/getCurrent.py -p $powerG -v $gridIR -o $current
rm -rf $powerG
ln $current /data/ToMarco/Yaswanth${core}/${1}${core}.current
ln $gridIR /data/ToMarco/Yaswanth${core}/${1}${core}.gridIR	
python /data/ToMarco/getV11.py -v $gridIR -o /data/ToMarco/Yaswanth${core}/v11
chmod 777 /data/ToMarco/Yaswanth${core}/*hmod 777 /data/ToMarco/Yaswanth${core}/*
