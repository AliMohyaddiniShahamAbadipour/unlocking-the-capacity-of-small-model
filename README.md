# unlocking-the-capacity-of-small-model
[Freeing the capacity of small models using data-driven artificial intelligence](https://drive.google.com/file/d/1IasZIyiJTidW2YO2Pv1H7hl2aqNkPvkk/view?usp=drive_link) is a new way to increase the quality of small models directly.
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
| Very small | [MASS-unsupNMT [Son19]](https://drive.google.com/file/d/1zMWUNMP_1cOSms64SeMhIRhiwJFVUd3g/view?usp=drive_link) | 0.97  | 1.16  | 1.04  |  1.31    |
| Very small | [Proposed Method](https://drive.google.com/file/d/1gms0fVEpcQssLKY5PRV62I7qzhOdoVFv/view?usp=drive_link) |  **6.45** | **7.72**  | **7**   |  **8.51**    |
|      small | [MASS-unsupNMT [Son19]](https://drive.google.com/file/d/1t2ArfzUB7CMTA1kDi5uDxXswF5PXosEq/view?usp=drive_link) | 9.86   |	8.06 |	11.5 |	8.14    |
|      small | [Proposed Method](https://drive.google.com/file/d/1VtXX5jR_s3JPhBk9mZQ8qmKDCQc9vrsW/view?usp=drive_link) |  **14.5**| **14**	|**17**	|**16.5**   |
|     Medium | [MASS-unsupNMT [Son19]](https://drive.google.com/file/d/1vvHbsQMmz5RtT3yv1r08bdMNnLI8izki/view?usp=drive_link) | 18.15	|17.75	|20.37	|20.33  |
|     Medium | [Proposed Method](https://drive.google.com/file/d/131Ka5jTGOxrBcVW0jLddqkej2-TVoEpi/view?usp=drive_link) |  **19.57**|**18.46**| **22.95**| **21.67** |
|     large  | MASS-unsupNMT [Son19] | 25	|23	|28	|28     |
|      large | Proposed Method       | 25	|23	|28	|28     |

## evaluation
Currently we implement unlocking-the-capacity-of-small-model for unsupervised NMT based on the codebase of [MASS](https://github.com/microsoft/MASS)

To run the code and see the results presented in the table, just run the following command in the codes folder. Before execution, it is necessary to put the appropriate test and evaluation data in the data path folder. Also, you should choose the dimensions of the model you use according to Table 1. The created translations are available in the output folder.

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
  --epoch_size 200000                                  \
  --max_epoch 30                                       \
  --eval_bleu true                                     \
  --eval_only true			\
  --dump_path  /output/ 
  --reload_model "$MODEL,$MODEL"                       \
```



