# MM2024-Virtual-Visual-Guided-Domain-Shadow-Fusion-via-Modal-Exchanging
The detailed description for Virtual Visual-Guided Domain-Shadow Fusion via Modal Exchanging

The steps for running our method: 1.1 bash data-process.sh; 1.2 bash data-train-large.sh (Fashion-MMT(large)); 1.3 bash data-checkpoint.sh; 1.4 bash data-generate.sh

In fairseq.tasks.translation.py, we manually set the address for visual features extracted from the pre-trained ResNet-101 model to avoid memory overflow issues, as the visual features are too large to handle efficiently. Additionally, please note that "data-bin-small-1," "data-bin-large-all," "EMMT," and "data-bin-2" represent the processed Fashion-MMT (clean), Fashion-MMT (large), EMMT, and Multi-30k datasets, respectively. If you have sufficient memory space, you can also extract visual features and incorporate them into the corresponding datasets. 

Detailed model configuration could be found in the paper.

If you have any questions, please feel free to contact me at the following email address: hzy23@stu.kust.edu.cn.
