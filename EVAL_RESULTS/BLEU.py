"""
SUN, Haoran 2024/11/24

This script evaluates the `flickr_test_2016` dataset for
machine translation using a pre-trained model and a set of custom weights.
Due to compatibility issues with the author's distributed training setup
and the code structure in `main.py` and `trainer.py`,
we are unable to directly use the provided scripts for evaluation.
Therefore, we have written this simplified Python script to generate translations and compute BLEU scores.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, model_name):
    """
    Load the pre-trained model and tokenizer, and initialize the model with custom weights.

    Args:
        checkpoint_path (str): Path to the custom model weights.
        model_name (str): Path to the pre-trained model directory.

    Returns:
        model: The initialized model loaded with custom weights.
        tokenizer: The tokenizer for text processing.
    """
    # Load the pre-trained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load custom weights
    state_dict = torch.load(checkpoint_path, map_location=device)  # Map weights to the correct device
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer


def translate_sentences(model, tokenizer, source_sentences, source_lang="en_XX", target_lang="fr_XX", max_length=50):
    """
    Generate translations for a list of source sentences using the model.

    Args:
        model: The translation model.
        tokenizer: The tokenizer for preprocessing.
        source_sentences (list): List of source sentences to translate.
        source_lang (str): Source language code (e.g., "en_XX").
        target_lang (str): Target language code (e.g., "fr_XX").
        max_length (int): Maximum length for the generated translation.

    Returns:
        translations (list): List of translated sentences.
    """
    translations = []
    tokenizer.src_lang = source_lang  # Set source language
    tokenizer.tgt_lang = target_lang  # Set target language

    for sentence in tqdm(source_sentences, desc="Translating sentences", unit="sentence"):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)

        # device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate translation
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=3
        )

        translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations.append(translated_sentence)
    return translations


def compute_bleu_score(hypotheses, references):
    """
    Compute the BLEU score between generated translations and reference translations.

    Args:
        hypotheses (list): List of generated translations.
        references (list): List of reference translations.

    Returns:
        float: The BLEU score.
    """
    bleu = corpus_bleu(hypotheses, [references])
    return bleu.score


def load_validation_data(source_file, reference_file):
    """
    Load source and reference sentences from files.

    Args:
        source_file (str): Path to the source language file.
        reference_file (str): Path to the reference language file.

    Returns:
        source_sentences (list): List of source sentences.
        reference_sentences (list): List of reference sentences.
    """
    # Load source sentences (e.g., English)
    with open(source_file, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f.readlines()]

    # Load reference sentences (e.g., French)
    with open(reference_file, 'r', encoding='utf-8') as f:
        reference_sentences = [line.strip() for line in f.readlines()]

    return source_sentences, reference_sentences


def save_results(translated_sentences, bleu_score, output_file="translations.txt", bleu_file="bleu_score.txt"):
    """
    Save the generated translations and BLEU score to files.

    Args:
        translated_sentences (list): List of translated sentences.
        bleu_score (float): The computed BLEU score.
        output_file (str): Path to save the translations.
        bleu_file (str): Path to save the BLEU score.
    """
    # Save translated sentences
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in translated_sentences:
            f.write(sentence + "\n")

    # Save BLEU score
    with open(bleu_file, "w", encoding="utf-8") as f:
        f.write(f"BLEU Score: {bleu_score}\n")


if __name__ == "__main__":
    checkpoint_path = "PATH/TO/CHECKPOINT"
    model_name = "PATH/TO/MBART"
    source_file = "PATH/TO/SOURCE_LANG"
    reference_file = "PATH/TO/REFERENCE_LANG"

    model, tokenizer = load_model(checkpoint_path, model_name)

    # Load the validation data
    source_sentences, reference_sentences = load_validation_data(source_file, reference_file)

    # Generate translations
    translated_sentences = translate_sentences(
        model, tokenizer, source_sentences, source_lang="en_XX", target_lang="fr_XX"
    )

    # Compute
    bleu_score = compute_bleu_score(translated_sentences, reference_sentences)
    print(f"Generated Translations: {translated_sentences[:5]}")  # Print the first 5 translations
    print(f"BLEU Score: {bleu_score}")

    save_results(translated_sentences, bleu_score, output_file="translations.txt", bleu_file="bleu_score.txt")
