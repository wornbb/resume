#!/bin/bash


echo "Hello, World!"
tick_per_cycle=500;
cycle_limit=350000;
#build/X86/gem5.opt \

#build/X86_MOESI_hammer/gem5.opt \
$x865/build/X86_MOESI_hammer/gem5.opt \
--outdir=$dump/ckp2c/ \
$fs/fs.py \
--kernel=$tool/FS_util/binaries/x86_64-vmlinux-2.6.22.9.smp \
--disk-image=$tool/FS_util/disks/x86root-parsec.img \
--cpu-type=AtomicSimpleCPU \
--num-cpus=2 \
--mem-size='2GB' \
--checkpoint-dir=$dump/ckp2c \
--script=$x865/configs/boot/hack_back_ckpt.rcS \
