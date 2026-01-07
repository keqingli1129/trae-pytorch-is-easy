import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class TorchActor(nn.Module):
    def __init__(self, model_path="Qwen/Qwen2.5-0.5B-Instruct", checkpoint_path=None):
        super().__init__()
        print(f"Loading {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # We still use HF to load the architecture/initial weights
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype="auto"
        )
        
        # Optionally overwrite with a local PyTorch checkpoint
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            
    def save_checkpoint(self, path):
        """Save the model weights using standard PyTorch save."""
        print(f"Saving checkpoint to {path}...")
        torch.save(self.model.state_dict(), path)
        print("Checkpoint saved.")

    def load_checkpoint(self, path):
        """Load the model weights using standard PyTorch load."""
        print(f"Loading checkpoint from {path}...")
        state_dict = torch.load(path, map_location=self.model.device)
        self.model.load_state_dict(state_dict)
        print("Checkpoint loaded.")

    def forward(self, input_ids):
        """
        Standard PyTorch forward pass.
        Returns: logits (Batch, Seq_Len, Vocab_Size)
        """
        # Directly return logits so it feels like a native PyTorch layer
        return self.model(input_ids).logits

    def generate(self, prompt, return_log_probs=False):
        """
        Generates a response.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=128,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        states = outputs.sequences
        actions = states[:, input_len:]
        
        if return_log_probs:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            return actions, states, transition_scores
        
        return actions, states

    def get_log_probs(self, states, actions):
        """
        Computes log probabilities of the 'actions' given the 'states'.
        """
        # 1. Forward pass (uses our new forward that returns logits directly)
        logits = self(states)
        
        # 2. Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = states[:, 1:]
        
        # 3. Compute log_softmax (Standard PyTorch)
        all_log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # 4. Gather probabilities of the actual tokens
        token_log_probs = torch.gather(all_log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # 5. Return only the action part
        action_len = actions.shape[1]
        return token_log_probs[:, -action_len:]

if __name__ == "__main__":
    # Example Usage
    actor = TorchActor()
    
    # 1. Test Forward (Direct Logits)
    dummy_input = torch.tensor([[1, 2, 3]]).to(actor.model.device)
    logits = actor(dummy_input)
    print(f"Logits shape: {logits.shape}") # [1, 3, Vocab]
    
    # 2. Test Generation
    actions, states = actor.generate("Hello, how are you?")
    print(f"Generated response: {actor.tokenizer.decode(actions[0])}")