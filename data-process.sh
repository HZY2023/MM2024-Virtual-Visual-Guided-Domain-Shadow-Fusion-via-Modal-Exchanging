#!/bin/bash

python3 preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref train \
  --validpref valid \
  --testpref test \
  --nwordssrc 17200 \
  --nwordstgt 9800 \
  --workers 12 \
  --destdir /en-de \
#  --srcdict data-bin/en-de/test2016/dict.en.txt \
#  --tgtdict data-bin/en-de/test2016/dict.de.txt