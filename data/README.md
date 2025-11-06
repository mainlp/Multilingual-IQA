For the full data statements, please refer to our paper.

InQA+ Data Statement
Data Statement for InQA+ following Bender and Friedman (2018):

A. Curation Rationale
InQA+ is intended to be used as a multi-lingual evaluation dataset for the Indirect Question Answering task. We collect and translate extracted question-answer pairs from the opensubtitles v2018 (Lison and Tidemann, 2016) corpus that contains data from http://www.opensubtitles.org. It serves as a high-quality, hand-labelled test data resource.

B. Language Variety
InQA+ is available in three languages that we provide together with their ISO 639 language codes:
    - English (EN)
    - German (DE)
    - Bavarian (BAR)

C. Speaker Demographic
The data stems from subtitles of undisclosed movies, which makes the origin unknown. The Bavarian translation stems from a native speaker of the Central Bavarian dialect.

D. Annotator Demographic
The annotations for all languages were carried out by a native speaker of the Bavarian dialect within the age range of 25-64 years who lives in the southern region of Bavaria in Germany. The annotator has a Bachelor's degree in Computational Linguistics.

E. Speech Situation
The data stems from subtitles of undisclosed movies, which does not allow the extraction of more information beyond the movie genre and release year. We assume that the speech was scripted by professional screen writers.

F. Text Characteristics
The data is annotated with labels that denote the polarity of the indirect answer:
1 - Yes: A clear yes or all gradients of yes (including weaker forms, e.g., maybe yes)
2 - No: A clear no or all gradients of no.
3 - Conditional Yes: A yes that only holds if certain conditions are true.
4 - Neither Yes nor No: A neutral answer that lies in the middle of yes and no.
5 - Other : The sentence does not match the questions as an answer.
6 - Lacking Context: Without further context, the answer cannot be clearly categorized.



GenIQA Data Statement
Data Statement for GenIQA following Bender and Friedman (2018):

A. Curation Rationale
GenIQA is intended to be used as a multi-lingual training dataset for the Indirect Question Answering task. The data was generated and annotated by gpt-4o-mini and serves the purpose of comparing artificial data with hand-curated data.

B. Language Variety
GenIQA is available in three languages that we provide together with their ISO 639 language codes:
    - English (EN)
    - German (DE)
    - Bavarian (BAR)

C. Speaker Demographic
The data stems from gpt-4o-mini, which directly generated data in all three languages. Therefore, we have no information about possible speaker origins that provided the original pre-training data that was referenced during the generation process.

D. Annotator Demographic
The annotations for all languages were carried out by gpt-4o-mini itself.

E. Speech Situation
As the data was created by gpt-4o-mini, we do not have information about the question-answer pairs and which circumstances the model considered internally during the generation process.

F. Text Characteristics
The data is annotated with labels that denote the polarity of the indirect answer:
1 - Yes: A clear yes or all gradients of yes (including weaker forms, e.g., maybe yes)
2 - No: A clear no or all gradients of no.
3 - Conditional Yes: A yes that only holds if certain conditions are true.
4 - Neither Yes nor No: A neutral answer that lies in the middle of yes and no.
5 - Other : The sentence does not match the questions as an answer.
6 - Lacking Context: Without further context, the answer cannot be clearly categorized.