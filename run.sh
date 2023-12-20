#!/bin/bash

mkdir -p logs/$1
rm -rf logs/$1/*
bash runs/$1.sh 2> logs/$1/err.log 1> logs/$1/out.log


