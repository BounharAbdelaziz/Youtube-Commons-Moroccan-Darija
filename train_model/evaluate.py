import re
from nemo.collections.asr.metrics.wer import word_error_rate

# ----------------------------------------------------------------- #
# ----------------------       Metrics       ---------------------- #
# ----------------------------------------------------------------- #

def normalize_arabic_text(text):
    """
    Arabic text normalization:
    1. Remove punctuation
    2. Remove diacritics
    3. Eastern Arabic numerals to Western Arabic numerals

    Arguments
    ---------
    text: str
        text to normalize
    Output
    ---------
    normalized text
    """
    # Remove punctuation
    punctuation = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؛؟]'
    text = re.sub(punctuation, '', text)

    # Remove diacritics
    diacritics = r'[\u064B-\u0652]'  # Arabic diacritical marks (Fatha, Damma, etc.)
    text = re.sub(diacritics, '', text)
    
    # Normalize Hamzas and Maddas
    text = re.sub('پ', 'ب', text)
    text = re.sub('ڤ', 'ف', text)
    text = re.sub(r'[آ]', 'ا', text)
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'[ئ]', 'ي', text)
    text = re.sub(r'[ء]', '', text)   

    # Transliterate Eastern Arabic numerals to Western Arabic numerals
    eastern_to_western_numerals = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)

    return text.strip()

def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    predictions_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    references_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    references = [normalize_arabic_text(r) for r in references_str]
    predictions = [normalize_arabic_text(p) for p in predictions_str]

    assert len(references) == len(predictions), "Predictions and references must have same length!"
    
    wer_percentage = word_error_rate(predictions, references) * 100
    cer_percentage = word_error_rate(predictions, references, use_cer=True) * 100
    
    return {
        'wer': wer_percentage,
        'cer': cer_percentage,
    }