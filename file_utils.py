

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
