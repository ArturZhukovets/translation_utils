from torchmetrics.functional import bleu_score
from torchmetrics.functional import chrf_score
from torchmetrics.functional import char_error_rate
from hlepor import single_hlepor_score
from evaluate import load


class MetricComparator:
    def __init__(self, preds: str, target: str):
        """
        :param preds: prediction line
        :param target: target (reference) line
        """
        self.preds = preds
        self.target = target
        self.n_gram = 4

    def bleu_compare(self) -> float:
        """Counts the bleu score by line"""
        splitted_line = self.preds.split()
        if len(splitted_line) <= 4:
            self.n_gram = 1
        elif len(splitted_line) <= 18:
            self.n_gram = 2
        elif len(splitted_line) <= 30:
            self.n_gram = 3
        score = bleu_score(self.preds.lower(), [self.target.lower()], n_gram=self.n_gram)
        return round(float(score), 3)

    def chrf_compare(self) -> float:
        """Calculates the CHRF++ score for the specified string"""
        score = chrf_score(self.preds.lower(), [self.target.lower()])
        return round(float(score), 3)

    def char_error_rate_compare(self) -> float:
        """
        This value indicates the percentage of
        characters that were incorrectly predicted.
        The lower the value, the better the performance of the ASR system
        with a CharErrorRate of 0 BEING A PERFECT SCORE.
        """
        score = char_error_rate(self.preds.lower(), [self.target.lower()])
        return round(float(score), 3)

    def hlepor_score_compare(self):
        """
        Calculate hLepor score on one pair of sentences.
        """
        hlepor_score = single_hlepor_score(
            hypothesis=self.preds,
            reference=self.target,
            preprocess=str.lower,
            separate_punctuation=True,
            language="russian"  # Удалить этот параметр если язык не русский
        )
        return round(hlepor_score, 3)


def bert_counter(preds: list[str], target: list[str]):
    """
    By default, using 'bert-base-multilingual-cased' if lang = "ru"
    :param preds: (list of str), list of sentences in translated text;
    :param target: (list of str), list of sentences in references text.
    """
    bertscore = load("bertscore")
    scores = bertscore.compute(predictions=preds, references=target, lang="ru")
    results = scores['f1']
    return results

