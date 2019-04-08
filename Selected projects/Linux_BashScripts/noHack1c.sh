#!/bin/bash


echo "Hello, World!"
tick_per_cycle=500;
cycle_limit=350000;
#build/X86/gem5.opt \

#build/X86_MOESI_hammer/gem5.opt \
$x865/build/X86_MOESI_hammer/gem5.opt \
--outdir=$dump/$1 \
$fs/fs_pdump.py \
--kernel=$tool/FS_util/binaries/x86_64-vmlinux-2.6.22.9.smp \
--disk-image=$tool/FS_util/disks/x86root-parsec.img \
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
--num-cpus=1 \
--mem-size='2GB' \
--dumpcycles=1 \
--ruby \
