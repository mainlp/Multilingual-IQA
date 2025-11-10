# Indirect Question Answering in English, German and Bavarian: A Challenging Task for High- and Low-Resource Languages Alike

- We present **InQA+** and **GenIQA**.

- **InQA+** is a multi-lingual extension of [IndirectQA](https://github.com/mainlp/indirectQA) in English, Standard German and Bavarian. It consists of indirect question-answer pairs from parallel movie scripts in English and German from [opensubtitles v2018 (Lison and Tiedemann, 2016)](https://opus.nlpl.eu/OpenSubtitles/en&de/v2024/OpenSubtitles) and hand-translated Bavarian sentences.

- **GenIQA** is a multi-lingual artificial dataset consisting of LLM-generated indirect question-answer pairs. 

- We train and evaluate multi-lingual transformer-based models (mBERT (Devlin et al., 2019), mDeBERTa (He et al., 2020) and XLM-R (Conneau et al., 2020)).

- We find that the IQA performance is poor in high- (English, German) and low-resource languages (Bavarian) and that a large training data amount is important. Further, GPT-4o-mini does not possess enough pragmatic understanding to solve the task well in any of the three tested languages.


### Corpus Statistics
![corp_stats](https://github.com/mainlp/Multilingual-IQA/assets/92130844/f635b03d-3433-4a05-ac07-d9a76a5d21c4)

### How to use this repository?
All subfolders containing data in `data` and `predictions` are in zip archives with the password `MaiNLP` so as to prevent potential inclusion in web-scraped datasets (cf. [Jacovi et al., 2023](https://aclanthology.org/2023.emnlp-main.308/)).

- `code`:
  - `data` contains dataset-related code, including the processing and filtering of the raw opensubtitles v2018 (Lison and Tiedemann, 2016) data and data perturbation.
  - `llms` contains LLM-related code, including the LLM testing and GenIQA generation code.
  - `train_predict` contains training- and classification-related code.

- `data`: contains zipped data files and a README with shortened data statements.

- `predictions`: contains the predictions, evaluation report and confusion matrix for each experiment, sorted by model (mBERT, mDeBERTa and XLM-R).

### Paper
If you use the data and/or code in this repository, please cite the following:
```
@inproceedings{winkler-2026-indirect,
  title = "Indirect Question Answering in English, German and Bavarian: A Challenging Task for High- and Low-Resource Languages Alike",
  author = "Winkler, Miriam and Blaschke, Verena and Plank, Barbara",
  year = "2026",
  booktitle = TODO,
  publisher = TODO,
}
```

### Acknowledgement
We thank the anonymous reviewers as well as the members of the MaiNLP research lab for their feedback, especially Rob van der Goot and Felicia Körner.

This work is supported by ERC Consolidator Grant DIALECT no. 101043235.

