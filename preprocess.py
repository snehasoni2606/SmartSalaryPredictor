import pandas as pd

def preprocess_input(experience, education, company, city):
    input_df = pd.DataFrame({
        'Experience': [experience],
        'Education': [education],
        'Company': [company],
        'City': [city]
    })
    return input_df
