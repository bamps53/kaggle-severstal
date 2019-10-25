#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CLS_CONFIG=$1
SEG_CONFIG=$2

$PYTHON split_folds.py --config $CLS_CONFIG
$PYTHON main_cls.py --config $CLS_CONFIG
$PYTHON main_seg.py --config $SEG_CONFIG
$PYTHON validation.py --cls_config $CLS_CONFIG --seg_config $SEG_CONFIG
$PYTHON prediction.py --cls_config $CLS_CONFIG --seg_config $SEG_CONFIG
