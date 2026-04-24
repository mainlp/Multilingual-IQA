For the full data statements, please refer to our paper.

# InQA+ Data Statement
Data Statement for InQA+ following Bender and Friedman (2018):

### A. Curation Rationale
This dataset extends the work of [Müller et al. (2024)](https://aclanthology.org/2024.lrec-main.791/) and adds a new Indirect Question Answering (IQA) resource. It contains 438 indirect question-answers per language in English, Standard German and Bavarian, a German dialect. It serves as an evaluation dataset for IQA. We use data from the opensubtitles v2018 corpus [(Lison and Tiedemann, 2016](https://aclanthology.org/L16-1147/). The subtitles stem from the website [opensubtitles.org](http://www.opensubtitles.org/).

The dataset consists of questions and indirect answers from movie scripts with numeric labels indicating the polarity of the indirect answers. They are separated by tabspaces with one question-answer pair and label per line.

#### Labels
The data is annotated with labels that denote the polarity of the indirect answer:
1. Yes: A clear yes or all gradients of yes (including weaker forms, e.g., maybe yes)
2. No: A clear no or all gradients of no.
3. Conditional Yes: A yes that only holds if certain conditions are true.
4. Neither Yes nor No: A neutral answer that lies in the middle of yes and no.
5. Other : The sentence does not match the questions as an answer.
6. Lacking Context: Without further context, the answer cannot be clearly categorized.

### B. Language Variety and Annotator Demographics
InQA+ is available in three languages that we provide together with their ISO 639 language codes:
- English (EN)
- German (DE)
- Bavarian (BAR)

The Bavarian translation depicts the dialectal variant of the border area between rural Upper and Lower Bavaria. The translations were carried out by a native speaker of Standard German and Bavarian in her 20s.

### C. Linguistic Situation
The English and German language as it occurs in the dataset is scripted, as it stems from movie scripts of various genres, e.g., comedy or crime. The Bavarian translation aims at being as natural sounding in the dialect as possible.

### Limitations
The Bavarian translation reflects the dialect of a single speaker.

### Majority Class Baseline Scores
"Yes" (Label 1) is the majority label in all datasets.

Majority Class Baseline InQA+
- Accuracy: 41.78
- F1: 9.82

Majority Class Baseline InQA+ REMAPPED
- Accuracy: 45.89
- F1: 15.73

Majority Class Baseline InQA+ YESNO
- Accuracy: 64.66
- F1: 39.27


# GenIQA Data Statement
Data Statement for GenIQA following Bender and Friedman (2018):

### A. Curation Rationale
This dataset is a training dataset for Indirect Question Answering (IQA) resource. It contains 1.500 LLM-generated indirect question-answer pairs per language in English, Standard German and Bavarian, a German dialect. The generation was performed with GPT-4o-mini in the time period of March to July 2025.

The label set is the same as for InQA+.

### B. Language Variety 
GenIQA is available in three languages that we provide together with their ISO 639 language codes:
- English (EN)
- German (DE)
- Bavarian (BAR)

### C. Linguistic Situation
The language in the dataset is scripted, as the LLM generations do not depict natural text. The generations for English and Standard German are high-quality. The quality of the Bavarian dialect is lacking (please refer to our paper for more details on the generated dialect quality).
