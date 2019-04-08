#!/usr/bin/zsh

cd $dump
base2c=$dump/${1}2c
work=${base2c}m5
echo $work
m5stat=$work/stats.txt
m5conf=$work/config.json
#templ=$template/mcpat-ruby_22nm_v13_steady_1cycle.xml
templ=$template/mcpat-ruby_22nm_v13_steady.xml
xml=${work: : -2}xml
op=${xml: : -3}.op
ptls=${op: : -3}.ptrace_loss
pt2c=${op: : -3}.ptrace
gridIR=${base2c}.gridIR
vtrace=${base2c}.vtrace
powerG=${base2c}.powerG
current=${base2c}.current
#Cleaning
rm -rf $dump/${1}.vtrace $dump/${1}.gridIR $dump/${1}.current $dump/${1}.powerG
rm -rf $gridIR $vtrace $powerG $current power_grid.test
timeout 30m $auto/g5script/run_bench.sh ${1}
python $tool/Dev-Toolbox/GEM5ToMcPAT.py -r -d $xml $m5stat $m5conf $templ > $work/log
$tool/mcpat-riscv-old/mcpat -infile $xml/mcpat-out.xml -print_level 5 -sim 1 -out_switch 1 -is_tdp 2 > $op
python $tool/Dev-Toolbox/McPATToHotSpot.py $op -o $pt2c
timeout 2h $software/voltspot20/voltspot  -c $software/voltspot20/pdn4c.config -f $des/vp/chips/22nm/Penryn22_ruby_ya_2c_v13.flp -p $pt2c -v $vtrace -gridvol_file $gridIR > buffer
mv ${gridIR}powerG $powerG
yidata=/data/yi/vioPred/data/${1}2c
mkdir $yidata
ln $gridIR $yidata/${1}2c.gridIR

