# Surveying public opinion using label prediction on social media data

**Abstract**: In this study, a procedure is proposed for surveying public opinion from big social media domain-specific textual data to minimize the difficulties associated with modeling public behavior. Strategies for labeling posts relevant to a topic are discussed. A two-part framework is proposed in which semiautomatic labeling is applied to a small subset of posts, referred to as the ”seed” in further text. This seed is used as bases for semisupervised labeling of the rest of the data. The hypothesis is that the proposed method will achieve better labeling performance than existing classification models when applied to small amounts of labeled data. The seed is labeled using posts of users with a known and consistent view on the topic. A semi-supervised multiclass prediction model labels the remaining data iteratively. In each iteration, it adds context-label pairs to the training set if softmax-based label probabilities are above the threshold. The proposed method is characterized on four datasets by comparison to the three popular text modeling algorithms (n-grams + tfidf, fastText, VDCNN) for different sizes of labeled seeds (5,000 and 50,000 posts) and for several label-prediction significance thresholds. Our proposed semi-supervised method outperformed alternative algorithms by capturing additional contexts from the unlabeled data. The accuracy of the algorithm was increasing by (3-10%) when using a larger fraction of data as the seed. For the smaller seed, lower label probability threshold was clearly a better choice, while for larger seeds no predominant threshold was observed. The proposed framework, using fastText library for efficient text classification and representation learning, achieved the best results for a smaller seed, while VDCNN wrapped in the proposed framework achieved the best results for the bigger seed. The performance was negatively influenced by the number of classes. Finally, the model was applied to characterize a biased dataset of opinions related to gun control/rights advocacy. The proposed semi-automatic seed labeling is used to label 8,448 twitter posts of 171 advocates for guns control/rights. On this application, our approach performed better than existing models and it achieves 96.5% accuracy and 0.68 F1 score.

```
@inproceedings{stanojevic2019surveying,
  title={Surveying public opinion using label prediction on social media data},
  author={Stanojevic, Marija and Alshehri, Jumanah and Obradovic, Zoran},
  booktitle={2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)},
  pages={188--195},
  year={2019},
  organization={IEEE}
}
```

# Datasets:
**News**: Public news datasets are published with prevoius studies as cited in the paper and can be found here: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M.
For the purpose of our research, as our model is trying to label a lot of unlabeled data using small amount of labeled data, we split labels from big parts of this dataset (see paper for more details). In order to reproduce our study without much effort, please start with datasets versions given here: https://drive.google.com/file/d/1IAnPQmT-XzPS8qxFCj65KMVxqVo9pItM/view?usp=sharing. You'll need to change the paths in the code. Also, since unlabeled and training files are updated with training, make sure to copy those files in separate folder for each training.

**Social Media**: Unfortunately, those data are not ours to share, but we are sharing here dataset which contains links of the tweets that we used. You can use Twitter API to download those tweets. Data were extracted using query based on hashtags provided by experts between 04/28/2017 - 06/02/2017 and between 02/06/2018 - 05/30/2018 Query was:
```
 "#iwillmarch" OR "#marchforourlives" OR "#whatif" OR "#guncontrol" OR "#guncontrolnow" OR "#neveragain" OR "#ifIDieinaSchoolShooting" OR "#NoRA" OR "#VoteThemOut" OR "#lethallegislation" OR "#gunreform" OR "#gunreformnow" OR "#enoughisenough" OR "#MSDstrong" OR "#SantaFestrong" OR "#EndGunViolence" OR "#NationalDieInDay" OR "#WearOrange" OR "#NoNotoriety" OR "#BanBumpStocks" OR "#AllowtheVote" OR "#OrangeCaps" OR "#NoNRAmoney"
 ```
