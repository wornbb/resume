#!/bin/bash


echo "Hello, World!"
tick_per_cycle=500;
cycle_limit=350000;
#build/X86/gem5.opt \

#build/X86_MOESI_hammer/gem5.opt \
$x865/build/X86_MOESI_hammer/gem5.opt \
--outdir=$dump/${1}m5 \
$fs/fs_pdump.py \
--script=$rcs/fluidanimate_2c_simsmall.rcS \
--kernel=$M5_PATH/binaries/x86_64-vmlinux-2.6.22.9.smp \
--disk-image=$M5_PATH/disks/x86root-parsec.img \
--cpu-type=DerivO3CPU \
--sys-clock='3.7GHz' \
--cpu-clock='3.7GHz' \
--sys-voltage='0.8V' \
--l1i_size=32kB \
--l1d_size=32kB \
--l2_assoc=8 \
--caches \
--l2cache \
--l2_size=2MB \
--num-cpus=2 \
--mem-size='2GB' \
--dumpcycles=1 \
--ruby \
