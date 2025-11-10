import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
import argparse

# opensubtitles data is not included in the repository and can be accessed at https://opus.nlpl.eu/OpenSubtitles/en&de/v2024/OpenSubtitles

parser = argparse.ArgumentParser(
    prog="data-process-raw-opus-german.py", 
    description="A program to parse the raw data of the opensubtitles v2018 corpus to extract questions sorted by genre. This data is used to assign extracted question-answer pairs to their respective genres."
    )

parser.add_argument("raw_data_dir", help="Path to the raw opensubtitles data directory.")
parser.add_argument("output_file", help="Path to the output file.")

args = parser.parse_args()

raw_dir = args.raw_data_dir
# not included in the package due to the large size
# raw data available available at https://opus.nlpl.eu/OpenSubtitles/en&de/v2024/OpenSubtitles

# save extracted data in various dictionaries
genre_to_questions = defaultdict(list)

# iterate over raw directory
# xlm parse code source: https://docs.python.org/3/library/xml.etree.elementtree.html
for path, _, file_names in os.walk(raw_dir):
    if file_names:
        for file_name in file_names:
            file_path = os.path.join(path, file_name)

            try:
                # parse xml file
                tree = ET.parse(file_path)
                root = tree.getroot()

            except:
                # skip files with malformed xml
                continue

            else:
                # extract genre(s) of file
                # file contents are appended to each genre, so there are duplicates between them
                genre_tag = root.find("./meta/source/genre")
                
                if genre_tag is not None: # check whether genre tag exists
                    genres = genre_tag.text.split(",")

                else:
                    genre = ["Unknown"]

                # extract sentences
                questions = []
                for s in root.findall("s"):
                    text = s.itertext()
                    sent = "".join(text).strip()

                    if sent.endswith("?"): # filter for questions
                        questions.append(sent)

                # save parsed data                
                for genre in genres:
                    # only save comedy and crime
                    if genre == "Comedy" or genre == "Crime": 
                        genre_to_questions[genre] += questions

# Dump extracted data into JSONs
output_path = args.output_file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(genre_to_questions, f, indent=2)


