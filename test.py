from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
from peft import PeftModel
new_checkpoint_path = "/model/adapter1"

try :
    print("Downloading mistal model from  HF")
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("HF Model Download completed, started merging with adapter ...")
    NEW_PEFT_MODEL = PeftModel.from_pretrained(base_model,
                                    new_checkpoint_path,
                                    torch_dtype=torch.float16,
                                    is_trainable=False,
                                    device_map="auto"
                                    )

    print("Saving merged model to /model/merged_mistralai_model")
    NEW_PEFT_MODEL = NEW_PEFT_MODEL.merge_and_unload(progressbar=True)
    
    NEW_PEFT_MODEL.save_pretrained("/model/merged_mistralai_model")

except Exception as msg:
    raise Exception("msg")



### HF_TOKEN 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

## zqiseam-ltifosforscsaws.registry.snowflakecomputing.com/nvidia_db/nvidia_schema/nvidia_image_repo