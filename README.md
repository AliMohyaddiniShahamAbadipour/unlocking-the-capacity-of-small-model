# unlocking-the-capacity-of-small-model
Freeing the capacity of small models using data-driven artificial intelligence is a new way to increase the quality of small models directly.
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

| Size of Model  | Method | Dataset | Dataset  | Dataset | Dataset |
| :-------------: | :-------------: | :-------------: | :-------------: |:-------------: | :-------------: |
| ----  | ---- | Valid en-fr |	Valid fr-en |	Test en-fr |	Test fr-en |
| Very small | MASS-unsupNMT [Son19] | 0.97  | 1.16  | 1.04  |  1.31    |
| Very small | Proposed Method       |  **6.45** | **7.72**  | **7**   |  **8.51**    |
|      small | MASS-unsupNMT [Son19] | 9.86   |	8.06 |	11.5 |	8.14    |
|      small | Proposed Method       |  **14.5**| **14**	|**17**	|**16.5**   |
|     Medium | MASS-unsupNMT [Son19] | 18.15	|17.75	|20.37	|20.33  |
|     Medium | Proposed Method       |  **19.57**	|**18.46**	|**22.95**	|**21.67**  |
|     large  | MASS-unsupNMT [Son19] | 25	|23	|28	|28     |
|      large | Proposed Method       | 25	|23	|28	|28     |
