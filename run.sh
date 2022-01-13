#!/bin/bash
for i in {1..5}
do
  python test.py -r 0.1 -a CMTN -d $i -p
done