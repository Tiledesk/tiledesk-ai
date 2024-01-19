import re
import json

def extract_entities(sentence):
    # Define regular expression patterns
    NORMAL_REGEX = "\[[\w+\s*]+\]\([\w+\s*]+\)"
    ENTITY_REGEX = "(\[[\w+\s*]+\])"
    ENTITY_NAME_REGEX = "\(([\w+\s*]+)\)"
    SYNONYM_REGEX = "\[[\w+\s*]+\]\{.+\}"
    DATA_REGEX = "\{.+\}"

    words = []
    labels = []

    syn_matches = list(re.finditer(SYNONYM_REGEX, sentence))

    for match in syn_matches:
        start, end = match.span()
        entity = match.group()
        before, _, after = sentence.partition(match.group())
        # Tokenize the words before and after the entity
        before_words = before.split()
        after_words = after.split()

        # Extract entity text and synonym text
        entity_text = re.search(ENTITY_REGEX, entity).group()[1:-1]
        synonym_text = re.search(DATA_REGEX, entity).group()

        try:
            synonym_data = json.loads(synonym_text)
        except json.JSONDecodeError as ex:
            raise ValueError(f"Error decoding synonym JSON: {synonym_text}")

        # Extract relevant attributes from synonym_data
        entity_name_text = synonym_data.get("entity")
        synonym_value = synonym_data.get("value")

        # Validate required attributes
        if entity_name_text is None or synonym_value is None:
            raise ValueError("Synonym data should have 'entity' and 'value' attributes")

        # Update the example with the entity text
        sentence = sentence[:start] + entity_text + sentence[end:]

        words.extend(before_words + [entity_text] + after_words)
        labels.extend(['O'] * len(before_words) + [entity_name_text] + ['O'] * len(after_words))

    return words, labels        




    for match in re.finditer(NORMAL_REGEX, sentence):
        start, end = match.span()
        entity = match.group()

        before, _, after = sentence.partition(match.group())
        # Tokenize the words before and after the entity
        before_words = before.split()
        after_words = after.split()

        entity_text = re.search(ENTITY_REGEX, entity).group()[1:-1]
        entity_name_text = re.search(ENTITY_NAME_REGEX, entity).group()[1:-1]
        sentence = sentence[:start] + entity_text + sentence[end:]

        words.extend(before_words + [entity_text] + after_words)
        labels.extend(['O'] * len(before_words) + [entity_name_text] + ['O'] * len(after_words))
        #print(sentence)
        #print(entity_text)
        #print(entity_name_text)

    return words, labels    

    
   
    

# Example sentence
sentence = "I'm working on [office hours](working_type)"
sentence1 = "I work on [office hour]{\"entity\": \"working_type\", \"value\": \"office hours\"}"

# Extract words and labels
words, labels = extract_entities(sentence1)

# Print words and labels
print("Words:", words)
print("Labels:", labels)
