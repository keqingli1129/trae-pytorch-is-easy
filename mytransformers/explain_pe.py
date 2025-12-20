import torch
import math

def explain_pe_logic():
    d_model = 8  # Small example
    print(f"--- Example with d_model = {d_model} ---")

    # 1. The specific line the user asked about
    arange_out = torch.arange(0, d_model, 2).float()
    print(f"\n1. torch.arange(0, {d_model}, 2).float():")
    print(arange_out)
    print(f"Shape: {arange_out.shape}")
    print("Explanation: Generates even indices [0, 2, 4, ...]. These correspond to the dimensions 2i.")

    # 2. The full div_term calculation (Frequency calculation)
    # Formula: 1 / (10000^(2i/d_model))
    # Log-space implementation: exp( -log(10000) * (2i / d_model) )
    div_term = torch.exp(arange_out * (-math.log(10000.0) / d_model))
    
    print(f"\n2. Full div_term (frequencies):")
    print(div_term)
    
    print("\n3. Verification against direct formula 1 / (10000^(2i/d_model)):")
    direct_formula = 1.0 / (10000.0 ** (arange_out / d_model))
    print(direct_formula)

if __name__ == "__main__":
    explain_pe_logic()