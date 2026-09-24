from enum import Enum


class TopicModelModes(Enum):
    """Topic modeling modes"""
    BERTOPIC = 'bertopic'
    LDA = 'lda'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return TopicModelModes[s.upper()]
        except KeyError:
            return s