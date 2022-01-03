import re

NUMBERS_AND_SPECIAL_CHARACTERS_REGEX = re.compile(r"[\W0-9]")
SINGLE_CHARACTERS_REGEX = re.compile(r"\s+[a-zA-Z]\s+")
SINGLE_CHARACTERS_FROM_START_REGEX = re.compile(r"\^[a-zA-Z]\s+")
MULTIPLE_SPACE_REGEX = re.compile(r"^b\s+")