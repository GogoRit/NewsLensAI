# evaluation/_utils.py

from transformers import PreTrainedTokenizerFast

def safe_truncate(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512
) -> str:
    """
    Truncate `text` to at most `max_length` tokens (including special tokens).
    Returns the decoded string.
    """
    # encode with truncation, then decode
    ids = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)