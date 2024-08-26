#!/bin/bash

python3 generate.py     data-bin-small-1 \
				--path results/mmtimg1/model.pt \
				--source-lang en --target-lang zh \
				--beam 5 \
				--num-workers 12 \
				--batch-size 128 \
				--results-path results \
				--remove-bpe \
#				--fp16 \