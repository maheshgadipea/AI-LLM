
from typing import Tuple, Union, List, Any
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import json
import torch
import tensorflow
import torch.nn.functional as F

class ModelTemplate():
    """
    This Class Demonstrate How To Implements ScoreBase Interface Class And It Basic Usage.
    """    
    def __init__(self):
        self.model_loaded = None
        self.model_name ="/app/"
        self.tokenizer_name ="roberta-large"
        self.loaded_dict ={11: 'kda_transactional',45: 'when_delta_on_share',48: 'when_share',
         25: 'topn_correlation',
         41: 'what',
         4: 'delta_on_share',
         50: 'which_contributing',
         24: 'topn_contribution_trend',
         39: 'trend_ratio',
         22: 'topn_contribution_to_growth',
         42: 'when',
         29: 'topn_growth_rate_trend',
         9: 'howmany_contribution',
         2: 'contribution_to_growth',
         7: 'how many',
         34: 'trend',
         21: 'share',
         36: 'trend_contribution_to_growth',
         5: 'forecast',
         16: 'list_growth_rate',
         0: 'Growth Rate',
         49: 'which',
         1: 'contribution',
         32: 'topn_share_trend',
         38: 'trend_growth_rate',
         28: 'topn_growth_rate',
         10: 'howmany_delta_on_share',
         31: 'topn_share',
         15: 'list_delta_on_share',
         8: 'howmany_contribute_to_growth',
         19: 'list_trend',
         47: 'when_ratio',
         37: 'trend_delta_on_share',
         14: 'list_contribution_to_growth',
         18: 'list_topn',
         20: 'ratio',
         33: 'topn_trend',
         46: 'when_growth_rate',
         23: 'topn_contribution_to_growth_trend',
         12: 'list',
         27: 'topn_delta_on_share_trend',
         53: 'why_share',
         26: 'topn_delta_on_share',
         51: 'why',
         35: 'trend_contribution',
         13: 'list_contribution',
         52: 'why_contribution',
         30: 'topn_ratio',
         3: 'correlation',
         40: 'trend_share',
         44: 'when_contribution_to_growth',
         43: 'when_contribution',
         17: 'list_share',
         6: 'geo_map'}
        if self.model_loaded is None:
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.loaded_model = AutoModelForSequenceClassification.from_pretrained(self.model_name,from_tf=True)
        
    
    def prediction_fn(self,
                      input_query
                       ):
        """
                Does the main prediction on pre_processed_input(Single Sample) using supplied model .
                :param pre_processed_input: Single Preprocessed Payload
                :return: Prediction Value From the model
                
                Important Notes:
                - Reshape your data array.reshape(1, -1) before predictions as it contains a single sample.
                    
        """
        
        query = input_query
        inputs = self.loaded_tokenizer(query,return_tensors="pt")
        outputs = self.loaded_model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        top_values, top_indices = torch.topk(probabilities, k=2, dim=1)
        finalDict = {
                    self.loaded_dict[top_indices.detach().numpy()[0][0]]:str(top_values.detach().numpy()[0][0]),
                    self.loaded_dict[top_indices.detach().numpy()[0][1]]:str(top_values.detach().numpy()[0][1])
                }
        finalDict = json.dumps(finalDict)
        return finalDict