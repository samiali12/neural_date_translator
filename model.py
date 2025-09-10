import numpy as np
import json
from keras.models import load_model

encoder_model = load_model('./attention_encoder_model.keras')
decoder_model = load_model('./attention_decoder_model.keras')
# print(model)


with open('params.json', 'r') as f:
    data = json.load(f)


Tx = data['Tx']
Ty = data['Ty']
human_vocab = data['human_vocab']
machine_vocab = data['machine_vocab']
human_vocab_size = data['human_vocab_size']
machine_vocab_size = data['machine_vocab_size']


def encode_text(input_text: str):
    input_text = input_text.lower()
    x = np.zeros((1, Tx, human_vocab_size))
    for i, ch in enumerate(input_text):
        x[0, i, human_vocab[ch]] = 1
    return x


def decode_sequence(input_text):
    inv_machine_vocab = {i: ch for ch, i in machine_vocab.items()}
    encoder_outputs, state_h, state_c = encoder_model.predict(input_text)
    target_seq = np.zeros((1, 1, len(machine_vocab)))
    target_seq[0, 0, machine_vocab["<start>"]] = 1.
    decoded_tokens = []
    for _ in range(Ty):
        preds, h, c = decoder_model.predict([target_seq, encoder_outputs, state_h, state_c])

        token_index = np.argmax(preds[0, -1, :])
        token = inv_machine_vocab[token_index]

        if token == "<end>":
            break
        decoded_tokens.append(token)

        # Prepare next input token
        target_seq = np.zeros((1, 1, len(machine_vocab)))
        target_seq[0, 0, token_index] = 1.

        state_h, state_c = h, c

    return "".join(decoded_tokens)
    

def predict(input_text):
    input_text = encode_text(input_text)
    result = decode_sequence(input_text)
    return result

