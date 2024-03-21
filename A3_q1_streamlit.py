import streamlit as st
import torch
from torch import nn
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Download Shakespeare dataset
shakespeare_path = "new_shakes.txt"

# Read the content of the downloaded file
with open(shakespeare_path, "r", encoding="utf-8") as file:
    text = file.read()

# Display the first 500 characters of the text

#text = text.lower()
text = text.strip()
text_list  = [txt for txt in text]
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(text))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['_'] = 0
itos = {i:s for s,i in stoi.items()}
#print(itos)
block_size = 10 # context length: how many characters do we take to predict the next one?
X, Y = [], []
context = [0] * block_size

for i in range(0, len(text)-10, block_size-1):
  #context = [0] * block_size
  for ch in text[i:i+block_size]:
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    #print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix] # crop and append

# Move data to GPU

X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)


class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x

@st.cache_data
def train_and_save_model(embedding_size_min, embedding_size_max):
    # Train the model here
    block_size = 10
    vocab_size = 61
    hidden_size = 10
    for embedding_size in range(embedding_size_min, embedding_size_max+1):
        model = NextChar(block_size, vocab_size, embedding_size, hidden_size)
        # Train the model

        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        import time
        # Mini-batch training
        batch_size = 4096
        print_every = 100
        elapsed_time = []
        # change number or epochs 
        for epoch in range(101):
            start_time = time.time()
            for i in range(0, X.shape[0], batch_size):
                x = X[i:i+batch_size]
                y = Y[i:i+batch_size]
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                opt.step()
                opt.zero_grad()
            end_time = time.time()
            elapsed_time.append(end_time - start_time)
            if epoch % print_every == 0:
                print(epoch, loss.item())
                
                # Save the trained model
        torch.save(model.state_dict(), f"model_emb{embedding_size}.pth")

#train_and_save_model(4, 5)  # Train models for embedding size 4 and 5


def generate_text(model, input_text, itos, stoi, block_size, length=100):
    context = [0] * block_size
    user_input_indices = [stoi[ch] for ch in input_text]

    if len(user_input_indices) > block_size:
        context = user_input_indices[-block_size:]
    else:
        context[block_size - len(input_text):] = user_input_indices
    text = ''
    for i in range(length):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        if ch == '_':
            break
        text += ch
        context = context[1:] + [ix]
    return text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    st.title("Next Character Prediction")

    embedding_size = st.slider("Embedding Size", min_value=4, max_value=5, value=4)
    num_characters = st.slider("Number of Characters to Generate", min_value=10, max_value=200, value=100)
    seed_text = st.text_input("Seed Text", "so and dead, till he shall fail his sister; and in true")
    #train_and_save_model(4, 5)
    train_and_save_model(4, 5)  # Train models for embedding size 4 and 5

    # Load the pre-trained model based on selected embedding size
    model = NextChar(block_size=10, vocab_size=61, emb_dim=embedding_size, hidden_size=10).to(device)
    model.load_state_dict(torch.load(f"model_emb{embedding_size}.pth"))
    model.eval()
    if st.button("Generate"):
        with st.spinner("Generating..."):
            # Load the pre-trained model
            #model = NextChar(block_size=10, vocab_size=61, emb_dim=embedding_size, hidden_size=10).to(device)
            #model.load_state_dict(torch.load(f"model_emb{embedding_size}.pth"))
            #model.eval()

            # Generate text
            generated_text = generate_text(model, seed_text, itos, stoi, block_size=10, length=num_characters)
            st.success("Generated Text:")
            st.write(generated_text)

if __name__ == "__main__":
    main()
