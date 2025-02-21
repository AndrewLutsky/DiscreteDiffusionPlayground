import torch
from model import DiscreteDiffusionTransformer

# Hyperparameters (same as training)
INPUT_DIM = 10  
MODEL_DIM = 64  
NUM_HEADS = 4  
NUM_LAYERS = 6  
VOCAB_SIZE = 100  

# Load model
model = DiscreteDiffusionTransformer(INPUT_DIM, MODEL_DIM, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE)
model.load_state_dict(torch.load("transformer_diffusion.pth"))
model.eval()

# Sample input for testing
test_input = torch.randint(0, VOCAB_SIZE, (1, INPUT_DIM))
test_target = torch.randint(0, VOCAB_SIZE, (1, INPUT_DIM))

with torch.no_grad():
    output = model(test_input, test_target)
    predicted = torch.argmax(output, dim=-1)

print("Input:", test_input)
print("Predicted Output:", predicted)

