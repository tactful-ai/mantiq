from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import random
import json
from typing import List, Dict, Callable, Any, Optional
import torch

def load_model(model_name: str, hf_token: str, quantized: bool = True):
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # Normalize float 4 quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for 4-bit base matrices
    bnb_4bit_use_double_quant=True  # Use nested quantization
    ) if quantized else None

    tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    use_fast= True
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically choose best device
        trust_remote_code=True,     # Required for some models
        token=hf_token
    )
    
    
    return model, tokenizer

class GenericDatasetProcessor:
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        self._dataset = None
        if random_seed is not None:
            random.seed(random_seed)

    def parse_dataset(self, dataset : List , parser_function: Callable, **kwargs) -> List[Dict[str, Any]]:
        return parser_function(dataset, **kwargs)

class DatasetParser:
    @staticmethod
    def default_parser(dataset: List[Dict[str, Any]], max_entries: Optional[int] = None, key_mapping: Optional[Dict[str, str]] = None):
        key_mapping = key_mapping or {"question": "question", "answer": "answer", "context": "context"}
        return [
            {
                "question": entry.get(key_mapping["question"]),
                "answer": entry.get(key_mapping["answer"]),
                "context": entry.get(key_mapping["context"]),
            }
            for entry in dataset[:max_entries]
        ]
