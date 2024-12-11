from tokenizer import Tokenizer

TRAINING_DATA = 'data/input.txt'

if __name__ == '__main__':
    tok = Tokenizer(input_file=TRAINING_DATA, vocab_size=4096)

    ids = tok.encode('Hello World!')
    print(ids)

    text = tok.decode(ids)
    print(text)