# python "F:\4json.py"
import numpy as np
import pandas as pd
# import ijson
import json
from pandas.io.json import json_normalize

# %%bash # use %%bash magic to print a preview of our file

# head ../input/roam_prescription_based_prediction.jsonl

# F:\gpt\nus sms corpus\smsCorpus_en_2015.03.09_all.json
# raw_data = pd.read_json("../input/roam_prescription_based_prediction.jsonl",
raw_data = pd.read_json("F:\gpt\\nus sms corpus\smsCorpus_en_2015.03.09_all.json",
                        lines=True,
                        orient='columns')
print(raw_data.shape)
raw_data.head()


# python "F:\4json.py"
