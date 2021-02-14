import torch
from transformers import LongformerModel, LongformerTokenizer
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1
# Attention mask values -- 0: no attention, 1: local attention, 2: global attention
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to global attention to be deactivated for all tokens
global_attention_mask[:, [1, 4, 21,]] = 1  # Set global attention to random tokens for the sake of this example
                                    # Usually, set global attention based on the task. For example,
                                    # classification: the <s> token
                                    # QA: question tokens
                                    # LM: potentially on the beginning of sentences and paragraphs
outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
sequence_output = outputs.last_hidden_state
pooled_output = outputs.pooler_output
print(outputs)