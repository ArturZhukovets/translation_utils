import requests
import os


class APIError(Exception):
    """Raised when request is not correct"""
    pass


class Translator:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_url = os.getenv("API_URL")

        headers = {
            "Authorization": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
            }
        self.session = requests.Session()
        self.session.headers = headers

    def get_text_translate(self, from_lang: str, to_lang: str, data: str) -> str:
        """
        Translate the text using json request
        :param from_lang: locale in format xx_XX
        :param to_lang: locale in format xx_XX
        :param data: text for translation
        :return: translated text
        """
        payload_dict = {
            "from": from_lang,
            "to": to_lang,
            "platform": "api",
            "data": data
        }
        response = self.session.post(self.api_url, json=payload_dict)
        if response.ok:
            translated_text = response.json()["result"]
        else:
            raise APIError(response.json()["err"])

        return translated_text
