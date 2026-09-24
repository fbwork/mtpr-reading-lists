import re

class REMatcher(object):
    def __init__(self, regexp):
        self.regexp = regexp
        self.rematch = None

    def match(self, match_string):
        if rematch := re.match(self.regexp, match_string):
            return rematch.groups()

def remove(text, needles):
    return replace(text, needles, '')

def replace(text, needles, dest=" "):
    for needle in needles:
        text = text.replace(needle, dest)

    return text

def reg_remove(text, needles):
    return reg_replace(text, needles, '')

def reg_replace(text, needles, dest=" "):
    for needle in needles:
        text = re.sub(needle, dest, text)
    return text

def split(value, joints, strip_chars=" "):
    value = value if type(value) is list else [value]

    for joint in joints:
        tmp = []
        for val in value:
            for v in val.split(joint):
                tmp.append(v.strip(strip_chars))
        value = tmp

    return value
