import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomerAutoencoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden_dim
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(
            self,
            input_seq
    ):
        """
        input_seq: Tensor of shape (batch_size, seq_len)
        """

        embedded = self.embedding(input_seq)    # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.encoder_lstm(embedded)  # hidden: (1, batch, hidden_dim)

        # Repeat hidden state for each time step of decoder input
        # Here we use the input again as a teacher-forced decoder input
        decoder_input = embedded  # for training, same as input sequence
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, torch.zeros_like(hidden)))
        logits = self.output_layer(decoder_output)  # (batch, seq_len, vocab_size)

        return logits  # You can apply softmax or cross-entropy loss outside
    

"""
# Example usage
model = CustomerAutoencoder(vocab_size=5, embed_dim=16, hidden_dim=32)
input_seq = torch.tensor([[0, 2, 3, 1]])  # batch of one sequence
output_logits = model(input_seq)

# Loss: reconstruction via cross-entropy
target_seq = input_seq  # for autoencoder, target is same as input
loss = F.cross_entropy(output_logits.view(-1, 5), target_seq.view(-1))
loss.backward()
"""


def get_customer_embeddings(model, input_seq):
    """
    Extracts the latent embeddings from the trained encoder of the model.
    
    Parameters:
        model: trained CustomerAutoencoder
        input_seq: Tensor of shape (batch_size, seq_len) with action IDs
        
    Returns:
        embeddings: Tensor of shape (batch_size, hidden_dim)
    """
    model.eval()  # Turn off dropout, etc.

    with torch.no_grad():
        embedded = model.embedding(input_seq)
        _, (hidden, _) = model.encoder_lstm(embedded)
        embeddings = hidden.squeeze(0)  # (batch_size, hidden_dim)
        
    return embeddings


"""
# Assume model is already trained
model = CustomerAutoencoder(vocab_size=5, embed_dim=16, hidden_dim=32)

# Example batch (2 users with 4-event histories)
input_seq = torch.tensor([
    [0, 1, 2, 3],
    [1, 1, 4, 0]
])

# Get embeddings
embeddings = get_customer_embeddings(model, input_seq)

print(embeddings.shape)  # ➜ torch.Size([2, 32])
print(embeddings)        # ➜ Your customer vectors
"""