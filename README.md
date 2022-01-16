# Knowformer

Source code, datasets for submission "_Position-aware Relational Transformer for Knowledge Graph Embedding_".

## Dependencies

* Python 3.7
* Python libraries: see requirements.txt

## Description

`./src` contains the source code of `Knowformer`, and the core element of the model implementation is in `./src/model`.

`./data` contains the datasets. Please `cd ./data` and `unzip` these datasets.

## Instruction

To test for link prediction on three datasets, please run `bash ./train_lp_fb15k237.sh`, `bash ./train_lp_wn18rr.sh` and `bash ./train_lp_yago.sh`, respectively. 

For entity alignment, please run `bash ./train_ea_zh.sh`, `bash ./train_ea_ja.sh` and `bash ./train_ea_fr.sh`, respectively. 
