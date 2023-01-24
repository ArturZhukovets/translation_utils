import logging
import csv
from translator_api import Translator
from metrics_service import MetricComparator, bert_counter
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format="%(message)s")


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

    with open("metrics.csv", 'w', encoding="utf-8") as f_o:
        writer = csv.writer(f_o)
        writer.writerow([
            "sent_prediction",
            "sent_reference",
            "chrf",
            "bleu",
            "cer(The lower the value then better)",
            "hlepor"
        ])
        for line_preds, line_reference in zip(text_prediction_ru.split("\n"), api_translated_text.split("\n")):

            comparator = MetricComparator(preds=line_preds, target=line_reference)
            chrf = comparator.chrf_compare()
            bleu = comparator.bleu_compare()
            cer = comparator.char_error_rate_compare()
            hlepor = comparator.hlepor_score_compare()

            writer.writerow([line_preds, line_reference, chrf, bleu, cer, hlepor])

            logging.info("#" * 20)
            logging.info(line_preds + "   ->   " + line_reference)
            logging.info(f"bleu - {bleu}")
            logging.info(f"CHRF++ - {chrf}")
            logging.info(f"CER - {cer}")
            logging.info(f"hLEPOR - {hlepor}")
            logging.info("#" * 20)
            logging.info("\n")

    """
    Здесь bert на CPU будет долго процессить. Можете теста ради запустить,
    но на небольшом количестве строк. Раскомментировать строки ниже
    """
    # bert_scores = bert_counter(preds=text_prediction_ru.split("\n"), target=api_translated_text.split("\n"))
    # logging.info(bert_scores)
