#!/usr/bin/zsh

cd $dump
input=$1
yiNN=$yi/nnData
work=$dump/${1}m5
echo $work
m5stat=$work/stats.txt
m5conf=$work/config.json
templ=/data/yi/ToolChain/Dev-Toolbox/templates/mcpat-ruby_22nm_v13_steady.xml
#templ=$template/mcpat-ruby_22nm_v13_steady.xml
xml=${work: : -2}xml
op=${xml: : -3}.op
pt=${op: : -3}.ptrace
viof=${op: : -3}.viomap
vios=${op: : -3}.viostat
#timeout 60s $auto/noHack2c.sh $1
#python $tool/Dev-Toolbox/GEM5ToMcPAT.py -r -d ${work: : -2}xml $m5stat $m5conf $templ > $work/log
#$tool/mcpat-riscv-old/mcpat -infile $xml/mcpat-out.xml -print_level 5 -sim 1 -out_switch 1 -is_tdp 2 > ${xml: : -3}.op
#python $tool/Dev-Toolbox/McPATToHotSpot.py $op -o ${op: : -3}.ptrace
$software/voltspot20/voltspot  -c $software/voltspot20/pdn.config -f $des/vp/chips/22nm/Penryn22_ruby_ya_2c_v13.flp -p $pt -v ${pt: : -7}.vtrace -gridvol_file ${pt: : -7}.gridIR -PDN_ptrace_warmup 100 -node_viotrace_file $viof -PDN_noise_th 5 -vio_file $vios> buffer
#rsync $viof $yiNN/${1}.viomap --remove-source-files 
#rsync ${pt: : -7}.gridIR $yiNN/${1}.viomap --remove-source-files
#rsync  $vios $yiNN/${1}.viostat --remove-source-files
