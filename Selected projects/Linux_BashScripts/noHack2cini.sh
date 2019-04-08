#!/bin/bash


echo "Hello, World!"
tick_per_cycle=500;
cycle_limit=350000;
#build/X86/gem5.opt \

#build/X86_MOESI_hammer/gem5.opt \
$x865/build/X86_MOESI_hammer/gem5.opt \
--outdir=$dump/ckp2cYasNoCPUvolt/ \
$fs/fs.py \
--kernel=$tool/FS_util/binaries/x86_64-vmlinux-2.6.22.9.smp \
--disk-image=$tool/FS_util/disks/x86root-parsec.img \
--num-cpus=2 \
--checkpoint-dir=$dump/ckp2cYasNoCPUvolt \
--script=$rcs/fluidanimate_2c_simsmall.rcS \
--sys-clock='3.7GHz' \
--cpu-clock='3.7GHz' \
--sys-voltage='0.7V' \
--num-cpus=2 \
--take-checkpoints=5214793040000,20000 \
--max-checkpoints=1 \





