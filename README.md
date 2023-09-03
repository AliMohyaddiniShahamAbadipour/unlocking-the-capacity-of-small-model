# unlocking-the-capacity-of-small-model
[Freeing the capacity of small models using data-driven artificial intelligence](https://drive.google.com/file/d/1HEuC63mg_zf041PwKg4xjGQIdzk3yZXO/view?usp=sharing) is a new way to increase the quality of small models directly.
While the general methods of compressing models to improve the quality of a small model are dependent on the creation and training of a larger model, the proposed method increases the quality of the model completely directly and independently of the creation and training of a larger model.
In general, the features of the proposed method can be listed as follows:
+ The proposed method is based on data and its implementation is very simple.
+ The proposed method is completely independent of larger models to improve model quality.
+ The proposed method frees the capacity of small models and does not improve the quality of the model in general.
+ The proposed method does not add any computational load to the model.

This method has been used to increase the quality of unsupervised neural machine translation on two languages, English and French. Four models have been created in different sizes, and the proposed method has improved the quality on three very small, small and medium models. The dimensions of the proposed models can be seen in the table below.

**Table 1: Features of four models used for Experiment**

| Size of Model  | Vocabulary size | Embedding Vector Dim | # layers  | # Heads | P-dropout    | Input output Embedding Sharing  | # Model parameters |
| :-------------: | :-------------: | :-------------: | :-------------: |:-------------: | :-------------: | :-------------: | :-------------: |
| Very small  |55 k   |128  | 1  |1  | 0.1  | yes  | 15.5M  | 
| small       | 65 k  |256  | 2  |2  |  0.1 | yes  | 41.5 M  | 
| medium      | 55 k  |420  | 4  |3  | 0.1  | yes  | 70.7 M| 
| large       | 65 k  |1024 | 6  |8  | 0.1  | yes  | 314.3 M  |
					
In the table below, the models trained with the proposed method as well as the usual method can be obtained.

**Tabel 2: BLEU value for the proposed method and the [Son19] method based on four evaluated datasets**

| Size of Model  | Model | Dataset | Dataset  | Dataset | Dataset |
| :-------------: | :-------------: | :-------------: | :-------------: |:-------------: | :-------------: |
| ----  | ---- | Valid en-fr |	Valid fr-en |	Test en-fr |	Test fr-en |
| Very small | [MASS-unsupNMT [Son19]](https://drive.google.com/file/d/1l-MgSPJyjsIvBbbShgh7lOZpUVwzWUpK/view?usp=sharing) | 0.97  | 1.16  | 1.04  |  1.31    |
| Very small | [Proposed Method](https://drive.google.com/file/d/12pPAlPZRkaIXzYueMjiAk5t0n9vxDi9w/view?usp=drive_link) |  **6.45** | **7.72**  | **7**   |  **8.51**    |
|      small | [MASS-unsupNMT [Son19]](https://drive.google.com/file/d/1t2ArfzUB7CMTA1kDi5uDxXswF5PXosEq/view?usp=drive_link) | 9.86   |	8.06 |	11.5 |	8.14    |
|      small | [Proposed Method](https://drive.google.com/file/d/1b-KfyyJtJJe3QnvKebl6bk4Z9vdj1UVU/view?usp=sharing) |  **14.5**| **14**	|**17**	|**16.5**   |
|     Medium | [MASS-unsupNMT [Son19]](https://drive.google.com/file/d/11l63fKPvJ9aomIrWODg0hkxqxrTWM5vx/view?usp=sharing) | 18.15	|17.75	|20.37	|20.33  |
|     Medium | [Proposed Method](https://drive.google.com/file/d/14kWuWOAhiVdjxRrU81xWcNWSHo7Tu_IL/view?usp=sharing) |  **19.57**|**18.46**| **22.95**| **21.67** |
|     large  | MASS-unsupNMT [Son19] | 25	|23	|28	|28     |
|      large | Proposed Method       | 25	|23	|28	|28     |

## Dependencies
Currently we implement unlocking-the-capacity-of-small-model for unsupervised NMT based on the codebase of [MASS](https://github.com/microsoft/MASS) The depencies are as follows:

+ Python 3
+ NumPy
+ PyTorch (version 0.4 and 1.0)
+ fastBPE (for BPE codes)
+ Moses (for tokenization)

## Data Ready
We use the BPE codes and vocabulary different from MASS. Here we take English-French as an example.
To select the VOCAB and CODES files according to the table one of the files in the [vocabulary&BPEcodes](vocabulary&BPEcodes) folder should be used

```
./get-data-nmt.sh --src en --tgt fr --reload_codes CODES --reload_vocab VOCAB

```

## Pre-training
Before running the codes, it is necessary to put the appropriate training data in the data path folder.
To generate training data, you must use the get-data-nmt.sh file that was introduced in the previous section.
Also, you should choose the dimensions of the model you use according to Table 1.

```
python train.py                                      \
--exp_name unsupMT_enfr                              \
--data_path ./data/processed/en-fr/                  \
--lgs 'en-fr'                                        \
--mass_steps 'en,fr'                                 \
--encoder_only false                                 \
--emb_dim 128                                       \
--n_layers 1                                         \
--n_heads 2                                          \
--dropout 0.1                                        \
--attention_dropout 0.1                              \
--gelu_activation true                               \
--tokens_per_batch 3000                              \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 100000                                  \
--max_epoch 1200                                      \
--eval_bleu true                                     \
--word_mass 0.5                                      \
--min_len 5                                          \
```

## Fine-tuning
After pre-training, we use back-translation to fine-tune the pre-trained model on unsupervised machine translation:

```
MODEL=pretrained.pth

python train.py \
  --exp_name unsupMT_enfr                              \
  --data_path ./data/processed/en-fr/                  \
  --lgs 'en-fr'                                        \
  --bt_steps 'en-fr-en,fr-en-fr'                       \
  --encoder_only false                                 \
  --emb_dim 128                                       \
  --n_layers 1                                         \
  --n_heads 2                                          \
  --dropout 0.1                                        \
  --attention_dropout 0.1                              \
  --gelu_activation true                               \
  --tokens_per_batch 2000                              \
  --batch_size 32	                                     \
  --bptt 256                                           \
  --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
  --epoch_size 100000                                  \
  --max_epoch 30                                       \
  --eval_bleu true                                     \
  --reload_model "$MODEL,$MODEL"                       \
```

## evaluation

To run the code and see the results presented in the table 2, just run the following command in the codes folder. Before execution, it is necessary to put the appropriate test and evaluation data in the data path folder. Also, you should choose the dimensions of the model you use according to Table 1. The created translations are available in the output folder.

```
MODEL=pp_vs.pth
python train.py \
  --exp_name unsupMT_enfr                              \
  --data_path  ./data/processed/en-fr/                  \
  --lgs 'en-fr'                                        \
  --bt_steps 'en-fr-en,fr-en-fr'                       \
  --encoder_only false                                 \
  --emb_dim 128                                       \
  --n_layers 1                                         \
  --n_heads 1                                          \
  --dropout 0.1                                        \
  --attention_dropout 0.1                              \
  --gelu_activation true                               \
  --tokens_per_batch 2000                              \
  --batch_size 32	                                     \
  --bptt 256                                           \
  --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
  --epoch_size 100000                                  \
  --max_epoch 30                                       \
  --eval_bleu true                                     \
  --eval_only true			\
  --dump_path  /output/ 
  --reload_model "$MODEL,$MODEL"                       \
```



