import re
import logging
import regex
from tileai.core.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

class WordTokenizer(Tokenizer):
    def __init__(self)->None:
       pass


    #Simple word tokenizer
    def tokenize(self, inp_str): ## This method is one way of creating tokenizer that looks for word tokens
        return re.findall(r"\w+", inp_str)

class StandarTokenizer(Tokenizer):
        def __init__(self)-> None:
            pass

        def tokenize(self, input_string):
            #FROM RASA
            words = regex.sub(
            # there is a space or an end of a string after it
            r"[^\w#@&]+(?=\s|$)|"
            # there is a space or beginning of a string before it
            # not followed by a number
            r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
            # not in between numbers and not . or @ or & or - or #
            # e.g. 10'000.00 or blabla@gmail.com
            # and not url characters
            r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
            " ",
            input_string,
            ).split()

            return words