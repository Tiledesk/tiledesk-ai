


#Simple word tokenizer
def tokenizer(self, inp_str): ## This method is one way of creating tokenizer that looks for word tokens
        return re.findall(r"\w+", inp_str)