import os
import glob
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

# TODO: refactor 
class Tokenizer:
    def __init__(self, input_file, vocab_size=4096, retrain=False):
        model = glob.glob("*.model")
        vocab = glob.glob("*.vocab")

        self.tokenizer_options = dict(
            input=input_file,
            input_format="text",
            model_prefix="tokenizer",
            model_type="bpe",
            vocab_size=vocab_size,
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            input_sentence_size=200_000_000,
            max_sentence_length=4192,
            seed_sentencepiece_size=1_000_000,
            shuffle_input_sentence=True,
            character_coverage=0.99995,
            byte_fallback=True,
            split_digits=True,
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_by_number=True,
            max_sentencepiece_length=16,
            add_dummy_prefix=True,
            allow_whitespace_only_pieces=True,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            num_threads=os.cpu_count(),
            minloglevel=10,
        )

        if retrain:
            for m in model:
                os.remove(m)
            for v in vocab:
                os.remove(v)

        if len(model) > 0 and len(vocab) > 0 and not retrain:
            print("Initializing tokenizer from file")

        else:
            print("Training tokenizer")
            SentencePieceTrainer.train(**self.tokenizer_options)

        self.sp = SentencePieceProcessor(glob.glob("*.model")[0])

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, tokens):
        return self.sp.decode(tokens)
    
    @property
    def pad_id(self):
        return self.sp.pad_id()
    
    @classmethod
    def from_file(file_path):
        pass
