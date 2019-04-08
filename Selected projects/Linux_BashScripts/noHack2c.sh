#!/bin/bash


echo "Hello, World!"
#build/X86/gem5.opt \

#$M5_PATH/binaries/x86_64-vmlinux-2.6.22.9.smp \
#build/X86_MOESI_hammer/gem5.opt \
$x865/build/X86_MOESI_hammer-modified/gem5.opt \
--outdir=$dump/${1}2cm5 \
$x865/configs/example/fs_16nm_freqmine.py \
--kernel=$M5_PATH/binaries/x86_64-vmlinux-2.6.22.9.smp \
--sys-clock='3.7GHz' \
--cpu-clock='3.7GHz' \
--sys-voltage='0.8V' \
--script=$x865/parsec-scripts/fluidanimate_2c_simsmall.rcS \
--num-cpus=2 \
--checkpoint-dir=$x865/analysis-ruby/boot_mem-atomic-nocache-check-simsmall/ \
--checkpoint-restore=1 \
--cpu-type=detailed \
--restore-with-cpu=timing \
--ruby \
--l1i_size=32kB \
--l1d_size=32kB \
--l2_size=2MB \
--topology=Mesh \
--mesh-rows=1 \
--mem-size='2GB' \
--rel-max-tick=530690232330 \
--maxinsts=100000 \
--warmup-insts=250000 \
--mem-channels=2 \
--num-dirs=4 \

