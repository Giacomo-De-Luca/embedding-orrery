from chromadb import Documents, EmbeddingFunction, Embeddings
import numpy as np 

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel



class EmbedTextQWEN(EmbeddingFunction[Documents]):

    def __init__(self, 
                model: str = 'Qwen/Qwen3-Embedding-0.6B', 
                device: str = 'mps', 
                batch_size: int = 12,
                task: str = 'Given a web search query, retrieve relevant passages that answer the query',
                max_length: int = 8192,
                normalize: bool = False,
                is_query: bool = False
                ) -> None:        
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.task = task
        self.device = device
        self.is_query = is_query

        self.tokeniser = AutoTokenizer.from_pretrained(model, padding_side='left')
        ## If Cuda is available, use Flash attention, else avoid it
        ## To note that I'm using fp32 for mps if we don't use flash attention
        ## So it's possible that we should let for cuda also let the possibility of using fp32
        ## But I should test how much the performance degrades with fp16/32
        self.model = AutoModel.from_pretrained(model, attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda() if device == 'cuda' else AutoModel.from_pretrained(model).to(device)
        self.model.eval()
        
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'


    def __call__(self, input: Documents) -> Embeddings:

        ## We only add instructions for queries
        if self.is_query:
            input = [self.get_detailed_instruct(task_description=self.task, query=text) for text in input]

        # Tokenize the input texts
        batch_dict = self.tokeniser(
            input,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch_dict.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])    

            if self.normalize == True: 
                ## I'm not sure if we need to normalize here or not
                ## since the visualiser may be more interesting in non normalised embedding space
                ## I leave the normalisation as an option
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
        ### Embedidng type wants a list of np.ndarray
        ### it returns a np array containing that contains other arrays
        ### we explictely convert to np.array first for the type checker

        return list(embeddings.detach().cpu().numpy())
            





