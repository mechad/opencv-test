#!/bin/bash
# ********************************************************
# @file: build.sh
# @author: Lin, Chao <chaochaox.lin@intel.com>
# @create time: 2024-03-16 08:49:55
# @last modified: 2024-03-16 08:49:55
# @description:
# ********************************************************

if [ $# != 1 ]; then
  echo $0 xx,cpp
  exit 1
fi

source_cpp=$(basename $1)
bin=${source_cpp%.*}

g++ -o $bin $1 `pkg-config --libs --static opencv4` -lzbar

