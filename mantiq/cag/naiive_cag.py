import os
import time
import torch
from transformers.cache_utils import DynamicCache
from ..cag.utils import load_model, GenericDatasetProcessor, DatasetParser

class NaiiveCAG:
    def __init__(self, model_name: str, hf_token: str, quantized: bool = True):
        """
        Initialize NaiiveCAG.
        Args:
            model_name: HuggingFace model name.
            hf_token: HuggingFace API token.
            quantized: Whether to load the model in quantized mode.
        """
        
        self.model, self.tokenizer = load_model(model_name, hf_token, quantized)
        self.processor = GenericDatasetProcessor()  # Initialize processor
        self.prompt_instruction = None

    def prepare_kvcache(self,dataset,saving_path, prompt_instruction: str, answer_instruction: str = None):
        """
        Prepare the KV cache using the JSON dataset specified at initialization.
        Args:
            prompt_instruction: Instruction for the system.
            answer_instruction: Instruction for the assistant.
        Returns:
            Tuple[DynamicCache, float]: KV cache and preparation time.
        """
        # Load and parse the dataset
        dataset = self.processor.parse_dataset(dataset ,parser_function=DatasetParser.default_parser)

        # Combine all contexts
        documents = "\n\n".join(
            f"Question: {entry['question']}\nAnswer: {entry['answer']}" + 
            (f"\nContext: {entry['context']}" if entry['context'] else "") +
            "\n"
            for entry in dataset
        )

        self.prompt_instruction = prompt_instruction or "You are an assistant for giving short answers."
        answer_instruction = answer_instruction or "Provide a concise and relevant answer."

        # Format the system prompt
        prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {self.prompt_instruction}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Context:
        ------------------------------------------------
        {documents}
        ------------------------------------------------
        
        {answer_instruction}
        """
        t1 = time.time()
        kv = self.preprocess_knowledge(prompt)
        
        saving_directory = saving_path + "/kv_cache.pt"
        
        self.write_kv_cache(kv, saving_directory)
        t2 = time.time()
        return kv, t2 - t1
    
    def prepare_kvcache_from_loader(self,loaded_data,saving_path, prompt_instruction: str, answer_instruction: str = None):
        """
        Prepare and save a key-value (KV) cache using the provided loaded data and instructions.

        This function generates a KV cache based on the provided `loaded_data`, formats it using 
        `prompt_instruction` and optional `answer_instruction`, and saves the result to a specified path.
        
        Args:
            loaded_data (str): The dataset or context loaded from a source to be used in the KV cache.
            saving_path (str): The directory path where the KV cache will be saved.
            prompt_instruction (str): Instruction or context for the system to format the prompt.
            answer_instruction (str, optional): Instruction for generating concise responses. Defaults to 
            "Provide a concise and relevant answer."
        Returns:
            Tuple[DynamicCache, float]: 
            - `DynamicCache`: The prepared KV cache object.
            - `float`: The time taken to prepare the KV cache, in seconds.
        """
        self.prompt_instruction = prompt_instruction or "You are an assistant for giving short answers."
        answer_instruction = answer_instruction or "Provide a concise and relevant answer."

        # Format the system prompt
        prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {self.prompt_instruction}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Context:
        ------------------------------------------------
        {loaded_data}
        ------------------------------------------------
        
        {answer_instruction}
        """
        t1 = time.time()
        kv = self.preprocess_knowledge(prompt)
        
        saving_directory = saving_path + "/kv_cache.pt"
        
        self.write_kv_cache(kv, saving_directory)
        t2 = time.time()
        return kv, t2 - t1

    def preprocess_knowledge(self, prompt: str) -> DynamicCache:
        """
        Preprocess knowledge and return KV cache.
        """
        embed_device = self.model.model.embed_tokens.weight.device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
        past_key_values = DynamicCache()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )
        return outputs.past_key_values

    def write_kv_cache(self, kv: DynamicCache, path: str):
        """
        Save the KV cache.
        """
        torch.serialization.add_safe_globals([DynamicCache, set])
        torch.save(kv, path)

    def read_kv_cache(self, path: str) -> DynamicCache:
        """
        Load the KV cache.
        """
        torch.serialization.add_safe_globals([DynamicCache])
        return torch.load(path, weights_only=True)

    def generate(self, input_query: str, past_key_values, max_new_tokens: int = 100) -> str:
        """
        Generate a response using the KV cache.
        """
        embed_device = self.model.model.embed_tokens.weight.device
        prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {self.prompt_instruction}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Question: {input_query}
        <|start_header_id|>assistant<|end_header_id|>
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        origin_ids = input_ids
        input_ids = input_ids.to(embed_device)

        output_ids = input_ids.clone()
        next_token = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
                next_token = next_token.to(embed_device)

                past_key_values = outputs.past_key_values

                output_ids = torch.cat([output_ids, next_token], dim=1)

                if next_token.item() in self.model.config.eos_token_id:
                    break
        return output_ids[:, origin_ids.shape[-1]:]
    
    
    def generate_response(self, query: str, past_key_values, max_new_tokens: int = 100) -> str:
        """
        Generate a response to the query.
        """
        output = self.generate(query, past_key_values, max_new_tokens)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return response