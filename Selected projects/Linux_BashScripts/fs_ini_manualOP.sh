#!/bin/bash


echo "Hello, World!"
tick_per_cycle=500;
cycle_limit=350000;
#build/X86/gem5.opt \

#build/X86_MOESI_hammer/gem5.opt \
build/X86/gem5.opt \
--outdir=./BS8827/ \
configs/example/fs_16nm_NoCycle.py \
--kernel=/data/yi/ToolChain/gem5-stable/full_system_images/x86-system/binaries/x86_64-vmlinux-2.6.22.9 \
--sys-clock='3.7GHz' \
--cpu-clock='3.7GHz' \
--sys-voltage='0.8V' \
--cpu-type=DerivO3CPU \
--l1i_size=32kB \
--l1d_size=32kB \
--caches \
--l2cache \
--l2_size=2MB \
--mem-size='2GB' \
--checkpoint-dir=$x865/BS8827 \
--script=./FS_util/test.rcS \
