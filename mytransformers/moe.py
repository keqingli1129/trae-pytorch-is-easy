import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()   
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x

class MoEGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts, bias=False)  # Replaced manual weights

    def forward(self, x):
        x_flat = x.view(-1, self.input_dim)
        gate_logits = self.gate(x_flat)  # Simpler forward pass
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_values = F.softmax(top_k_logits, dim=-1)
        return top_k_indices, top_k_values


class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=None, num_experts=4, top_k=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        
        if output_dim is None:
            output_dim = input_dim
            
        self.experts = nn.ModuleList([ExpertModule(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gating = MoEGating(input_dim, num_experts, top_k)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, x.shape[-1]) # (total_tokens, input_dim)
        
        # Get routing decisions
        # top_k_indices: (total_tokens, top_k) -> Which experts to use
        # top_k_values: (total_tokens, top_k) -> Weight for each selected expert
        top_k_indices, top_k_values = self.gating(x)
        
        # Initialize output container
        # We assume ExpertModule output_dim matches the initialized output_dim
        # Using the shape of the first expert's output to determine size
        dummy_out = self.experts[0](x_flat[:1])
        expert_output_dim = dummy_out.shape[-1]
        final_output = torch.zeros(x_flat.size(0), expert_output_dim, device=x.device)

        # Iterate through each of the k selected experts for every token
        for k in range(self.top_k):
            # Get the expert index and weight for the k-th choice for all tokens
            expert_idx = top_k_indices[:, k] # (total_tokens,)
            weight = top_k_values[:, k].unsqueeze(1) # (total_tokens, 1)

            # We need to process tokens expert by expert to batch computation
            for i, expert in enumerate(self.experts):
                # Find which tokens chose expert 'i' as their k-th choice
                mask = (expert_idx == i)
                if mask.any():
                    selected_input = x_flat[mask]
                    expert_out = expert(selected_input)
                    # Add weighted output to the final accumulator
                    # We use strict indexing to place results back in correct rows
                    final_output[mask] += weight[mask] * expert_out

        return final_output.view(batch_size, seq_len, -1)
