import re
import string


def lower(text):
    return text.lower()

def remove_citations(text):
    return re.sub(r'\[\s*(\d+\s*(,\s*\d+)*)\s*\]', '', text)

def process_response(response):
    if response == None:
        return None
    final_result = remove_citations(lower(response))
    if final_result.strip() == "":
        return None
    else:
        return final_result
    

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_citations(text):
        return re.sub(r'\[\s*(\d+\s*(,\s*\d+)*)\s*\]', '', text)

    final_result = white_space_fix(remove_articles(remove_punc(remove_citations(lower(s)))))
    if final_result == None:
        print("final_result is empty")
        return "a"
    else:
        return final_result
