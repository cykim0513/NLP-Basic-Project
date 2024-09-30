import re

def only_completion(self, dataframe):
    """
    완성형 한글 음절 추출
    """
    def clean_text(text):
        pattern = r"[^가-힣\s]"
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text
    for text_column in self.text_columns:
        dataframe[text_column] = dataframe[text_column].apply(clean_text)
        
    return dataframe