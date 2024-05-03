
from typing import Tuple, Union, List, Any
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import json
import torch
import tensorflow
import torch.nn.functional as F
import re

class ModelTemplate():
    """
    This Class Demonstrate How To Implements ScoreBase Interface Class And It Basic Usage.
    """    
    def __init__(self,model_path,adapter_path,adapter_name):
        print(model_path,
              adapter_path,
              adapter_name)
        
        self.loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.loaded_model.load_adapter(adapter_path)
        self.loaded_model.set_adapter(adapter_name)
        self.loaded_model.to("cuda")
        
        
    
    def generate_prompt(self,
                      input_query
                       ):
        
        messages = [
                {"role": "user",
                 "content": input_query
                }
            ]
        encoders = self.loaded_tokenizer.apply_chat_template(messages,return_tensors="pt")
        model_inputs = encoders.to("cuda")
        generated_ids = self.loaded_model.generate(model_inputs,max_new_tokens=100,do_sample=True)
        decoded = self.loaded_tokenizer.batch_decode(generated_ids)
        return re.sub(r"<s>\s*|\[INST\].*?\[/INST\]|</s>","",decoded[0])
