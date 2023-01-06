from translator_api import Translator
from torchmetrics.functional import bleu_score
from torchmetrics.functional import chrf_score
from dotenv import load_dotenv

load_dotenv()


def get_text_from_file(path_to_file: str) -> str:
    try:
        with open(path_to_file, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    except Exception:
        raise OSError("Error while writing to file")


def save_text_to_file(path_to_file: str, text: str) -> None:
    try:
        with open(path_to_file, 'w', encoding="utf-8") as file:
            file.write(text)

    except Exception:
        raise OSError("Error while writing to file")


def bleu_compare(prediction: str, reference: str, n_gram: int = 4) -> float:
    """Counts the bleu score by line"""
    return round(float(bleu_score(prediction.lower(), [reference.lower()], n_gram=n_gram)), 3)


def chrf_compare(prediction: str, reference: str) -> float:
    """Counts the CHRF++ score by line"""
    return round(float(chrf_score(prediction, [reference])), 3)


if __name__ == '__main__':

    text_for_translate = get_text_from_file("/home/user/Desktop/Texts_compares/source_en.txt")
    session = Translator()
    api_translated_text = session.get_text_translate(
        from_lang='en_US',
        to_lang="ru_RU",
        data=text_for_translate
    )

    text_prediction_ru = get_text_from_file("/home/user/Desktop/Texts_compares/prediction_ru.txt")
    # Save in file (uncomment if you need)
    # save_text_to_file(path_to_file: str, translated_text)

    for line_preds, line_reference in zip(text_prediction_ru.split("\n"), api_translated_text.split("\n")):

        chrf = chrf_compare(line_preds, line_reference)

        if len(line_preds.split()) == 1:
            bleu = bleu_compare(line_preds, line_reference, n_gram=1)
        elif len(line_preds.split()) <= 6:
            bleu = bleu_compare(line_preds, line_reference, n_gram=2)
        else:
            bleu = bleu_compare(line_preds, line_reference, n_gram=4)

        print("#" * 20)
        print(line_preds + "   ->   " + line_reference)
        print(f"bleu - {bleu}")
        print(f"CHRF++ - {chrf}")
        print("#" * 20)
        print()



