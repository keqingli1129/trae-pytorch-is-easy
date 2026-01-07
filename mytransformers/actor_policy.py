import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActorPolicy(nn.Module):
    def __init__(self, model_path="Qwen/Qwen2.5-0.5B-Instruct"):
        super().__init__()
        print(f"Loading {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype="auto"
        )
        
    def generate(self, prompt, return_log_probs=False):
        """
        Generates a response and returns separate actions and states.
        Interface: generate(prompt) -> actions, states
        If return_log_probs is True, returns actions, states, log_probs
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=128,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        states = outputs.sequences # Full sequence: [Prompt + Response]
        actions = states[:, input_len:] # Just the generated response tokens
        
        if return_log_probs:
            # Compute log probabilities for the generated tokens
            # transition_scores are the log probabilities of the generated tokens
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            # transition_scores shape: (batch_size, generated_len)
            # This matches the shape expected for log_probs corresponding to 'actions'
            return actions, states, transition_scores
        
        return actions, states

    def get_log_probs(self, states, actions):
        """
        Computes log probabilities of the 'actions' given the 'states'.
        """
        # Forward pass on the full sequence (states)
        # output.logits shape: [Batch, Seq_Len, Vocab]
        outputs = self.model(states)
        logits = outputs.logits
        
        # Shift logits and labels for next-token prediction
        # Logits at index t predict token at index t+1
        shift_logits = logits[:, :-1, :]
        shift_labels = states[:, 1:]
        
        # Compute log probabilities over the vocabulary
        all_log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        
        # Gather the log probs of the actual tokens that appeared in the sequence
        # gather expects index to have same dims as input except for the dimension to gather
        token_log_probs = torch.gather(all_log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # We only care about the log_probs for the 'actions' (the response part)
        # The actions correspond to the end of the sequence
        action_len = actions.shape[1]
        
        # Return only the log probs corresponding to the generated actions
        return token_log_probs[:, -action_len:]
        
    def forward(self, input_ids):
        return self.model(input_ids)