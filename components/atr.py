import html
import logging
from collections import Counter
from typing import List

import ahocorasick
from nltk.corpus import stopwords
from nltk.util import ngrams
from sortedcontainers import SortedList

import utils
from models import *
from utils import TaskAssistant
from .database import setup as database_setup

current_path = Path(__file__).parent
atr_output_path = current_path.joinpath('output', 'atr')
atr_resource_path = current_path.joinpath('resources', 'atr')
atr_log_path = Path(os.getenv('LOG_PATH')).joinpath('atr.log')
os.makedirs(atr_output_path, exist_ok=True)
os.makedirs(atr_resource_path, exist_ok=True)

erased_terms_file_path = atr_output_path.joinpath("erased_terms.bin")
technical_terms_file_path = atr_output_path.joinpath("technical_terms.bin")


def setup_logger():
    atr_log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=atr_output_path.joinpath('atr.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()


class ErasedTerms:
    STOPWORD = 'stop-word'
    COMMON_WORD = 'common-word'
    TRAILING_STOPWORD = 'trailing stop-word'
    SINGLE_PAPER = 'single-paper appearance'
    SUBSUMED = 'subsumed by a frequent gram'

    def __init__(self, stopwords=None, trailing_stopwords=None, common_words=None, single_paper=None, subsumed=None):
        self.stopwords = stopwords or []
        self.trailing_stopwords = trailing_stopwords or []
        self.common_words = common_words or []
        self.single_paper = single_paper or []
        self.subsumed = subsumed or []

    def reason(self, term):
        """
        Reveals the reason that a single gram is not present in the list of frequent grams
        :param term: The query term
        :return: The removal reason of a gram
        """
        if term in self.stopwords:
            return self.STOPWORD
        if term in self.common_words:
            return self.COMMON_WORD
        if term in self.trailing_stopwords:
            return self.TRAILING_STOPWORD
        if term in self.single_paper:
            return self.SINGLE_PAPER
        if term in self.subsumed:
            return self.SUBSUMED

        return "Infrequent grams"


def aggregate_grams(results, bar, grams: Counter, papers_contain_grams: Counter):
    _grams, _papers_contain_grams = results
    grams.update(_grams)
    papers_contain_grams.update(_papers_contain_grams)
    bar.update()


def clean_title(text):
    """
    Cleans a text (mostly paper titles) based on a few rules.
    :param text: Input, dirty text
    :return: Cleaned text
    """
    text = html.unescape(text)
    text = utils.remove(text, "\\\"'?!()’“”")
    text = text.strip(' \'"\n')
    text = utils.reg_remove(text, ["<.*>", "&.*;", "<i", "\/i>"])
    text = utils.reg_replace(text, ["\s{2,}"], " ")

    return utils.split(text.lower(), [':', '=', ';', '--', ' - ', ' – '])


def clean_gram(gram):
    """
    Cleans a gram based on a few rules.
    :param gram: Input, dirty gram
    :return: Cleaned gram
    """
    if len(gram) < 2:
        return ''

    return utils.reg_remove(gram, ["\s*[.\-–=,\/]\s*$", "^\s*[.\-–=,\/]\s*"])


def generate_n_grams(papers: List[Paper]):
    logger = setup_logger()
    grams = Counter()
    paper_contain_grams = Counter()
    for paper in papers:
        grams_in_paper = set()
        title_parts = clean_title(paper.title)
        logger.info(' <--> '.join(title_parts))
        for title in title_parts:
            tokens = [t for g in title.split() if (t := clean_gram(g))]
            if (n := len(tokens)) < 3:
                continue
            for i in range(1, n):
                for ngram_seq in ngrams(tokens, i):
                    if any([not val or val.isdigit() or '&' in val for val in ngram_seq]):
                        continue
                    sentence = ' '.join(ngram_seq)
                    grams.update([sentence])
                    if sentence not in grams_in_paper:
                        paper_contain_grams.update([sentence])
                        grams_in_paper.add(sentence)
    return grams, paper_contain_grams


def remove_single_paper_grams(grams: Counter):
    return {key for key, counter in grams.items() if counter == 1}


def remove_common_words(grams: Counter, c_words: set):
    return {key for key in grams if key in c_words}


def remove_stopwords(grams: Counter):
    stop_words = set(stopwords.words("english"))
    return {key for key in grams if key in stop_words}


def remove_trailing_stopwords(grams: Counter):
    stop_words = set(stopwords.words("english"))
    return {key for key in grams if key.split()[-1] in stop_words or key.split()[0] in stop_words}


def remove_subsumed_keys(grams: Counter, all_grams: Counter):
    logger = setup_logger()
    sorted_grams = SortedList(grams.keys(), key=lambda x: (len(x.split()), x))
    subsumed_keys = set()
    for i, child_gram in enumerate(sorted_grams):
        child_len = utils.gram_len(child_gram)
        child_reps = grams[child_gram]
        for parent_gram in sorted_grams[i + 1:]:
            if utils.gram_len(parent_gram) != child_len + 1:
                break
            if child_gram in parent_gram and child_reps < grams[parent_gram] * 1.30:
                subsumed_keys.add(child_gram)
                break
    logger.info(f"Identified {len(subsumed_keys)} subsumed grams")
    return subsumed_keys


def remove_selected_grams(keys: set, bar, erased: list, grams: Counter):
    erased.extend(keys)
    for key in keys:
        grams.pop(key, None)
    bar.update()


def load_common_words():
    logger = setup_logger()
    _common_words = set()
    resource_files = ['words.txt', 'twl98.txt', 'sowpod.txt']
    for file_name in resource_files:
        file_path = atr_resource_path.joinpath(file_name)
        if not file_path.is_file():
            logger.info(f"Warning: {file_name} not found in {atr_resource_path}. Skipping...")
            continue
        with open(file_path) as f:
            _common_words.update(line.strip(' \r\n').lower() for line in f)
    if not _common_words:
        logger.info("Warning: No common words loaded. Using empty set.")
    return list(_common_words)


def count_terms(papers: List[Paper], terms: List[TechnicalTerm]):
    automaton = ahocorasick.Automaton()
    for term in terms:
        automaton.add_word(term.text.lower(), term.id)
    automaton.make_automaton()
    results = []
    for paper in papers:
        if not paper.body or not isinstance(paper.body, str):
            continue
        body = paper.body.lower()
        freq = Counter()
        for end_index, term_id in automaton.iter(body):
            freq[term_id] += 1
        results.extend(
            PaperTechnicalTerm(paper=paper.id, technical_term=term_id, frequency=freq[term_id]) for term_id in freq)
    return results


def aggregate_paper_terms(partial_paper_terms, bar, terms):
    """
    Aggregates PaperTechnicalTerm records.
    :param partial_paper_terms: Recently generated PaperTechnicalTerm records
    :param bar: The singleton progressbar object
    :param terms: List of terms to be updated
    """
    terms += partial_paper_terms
    bar.update()


def generate_technical_terms(force: bool = False, chunk_size: int = 100):
    logger = setup_logger()
    papers = list(Paper.select())
    common_words = load_common_words()
    output_file = atr_output_path.joinpath('titles.txt')

    erased_terms = ErasedTerms()
    grams, papers_contain_grams = Counter(), Counter()
    TaskAssistant(
        generate_n_grams,
        {'papers': papers},
        "Generating grams from paper titles...",
        aggregate_grams,
        task_kwargs={'output_file': output_file},
        callback_kwargs={'grams': grams, 'papers_contain_grams': papers_contain_grams},
        chunk_size=chunk_size
    ).execute()
    logger.info(f"{len(grams)} grams are generated!")

    TaskAssistant(
        remove_single_paper_grams,
        {'grams': papers_contain_grams},
        "Removing single-paper appearance grams...",
        remove_selected_grams,
        callback_kwargs={'erased': erased_terms.single_paper, 'grams': grams}
    ).execute()
    logger.info(f"{len(erased_terms.single_paper)} single paper grams were removed!")

    TaskAssistant(
        remove_stopwords,
        {'grams': grams},
        "Removing stop-words...",
        remove_selected_grams,
        callback_kwargs={'erased': erased_terms.stopwords, 'grams': grams}
    ).execute()
    logger.info(f"{len(erased_terms.stopwords)} stop-words were removed!")

    TaskAssistant(
        remove_trailing_stopwords,
        {'grams': grams},
        "Removing trailing stop-word grams...",
        remove_selected_grams,
        callback_kwargs={'erased': erased_terms.trailing_stopwords, 'grams': grams}
    ).execute()
    logger.info(f"{len(erased_terms.trailing_stopwords)} trailing stop-word grams were removed!")

    TaskAssistant(
        remove_common_words,
        {'grams': grams},
        "Removing common words...",
        remove_selected_grams,
        callback_kwargs={'erased': erased_terms.common_words, 'grams': grams},
        task_kwargs={'c_words': set(common_words)}
    ).execute()
    logger.info(f"{len(erased_terms.common_words)} common words were removed!")

    TaskAssistant(
        remove_subsumed_keys,
        {'grams': grams},
        "Removing subsumed grams...",
        remove_selected_grams,
        callback_kwargs={'erased': erased_terms.subsumed, 'grams': grams},
        task_kwargs={'all_grams': grams}
    ).execute()
    logger.info(f"{len(erased_terms.subsumed)} subsuming grams were removed!")

    technical_terms = remove_infrequent_grams(grams, 0.25)

    try:
        with db.atomic():
            if force:
                TechnicalTerm.truncate_table(cascade=True)
            TechnicalTerm.bulk_create(
                [TechnicalTerm(text=text, frequency=frequency) for text, frequency in technical_terms.items()],
                batch_size=200
            )
    except Exception as e:
        logger.error(f"Error saving technical terms to database: {e}")
        raise

    return erased_terms, technical_terms


def remove_infrequent_grams(grams, percent=1.0):
    """
    Identify grams that are infrequent. The infrequency is checked in uni- and bi-grams separately.
    :param grams: List of grams in the form of <gram, frequency>
    :param percent: The percent of grams considered to be frequent in each of uni- and bi-grams.
    :return: Frequent n-grams
    """
    uni_grams = Counter({key: counter for key, counter in grams.items() if len(key.split()) == 1})
    bi_grams = Counter({key: counter for key, counter in grams.items() if len(key.split()) == 2})
    n_grams = Counter({key: counter for key, counter in grams.items() if len(key.split()) > 2})

    n_grams.update(Counter({key: counter for key, counter in uni_grams.most_common(round(percent * len(uni_grams)))}))
    n_grams.update(Counter({key: counter for key, counter in bi_grams.most_common(round(percent * len(bi_grams)))}))

    return n_grams


def count_technical_terms_in_papers(force: bool = False, chunk_size: int = 100):
    logger = setup_logger()
    paper_technical_terms = []
    TaskAssistant(
        count_terms,
        {'papers': list(Paper.select())},
        "Counting terms in paper texts...",
        aggregate_paper_terms,
        callback_kwargs={'terms': paper_technical_terms},
        task_kwargs={'terms': list(TechnicalTerm.select())},
        chunk_size=chunk_size
    ).execute()
    logger.info(f"{len(paper_technical_terms)} paper-technical term records generated!")

    try:
        with db.atomic():
            if force:
                PaperTechnicalTerm.truncate_table(cascade=True)
            PaperTechnicalTerm.bulk_create(paper_technical_terms, batch_size=200)
    except Exception as e:
        logger.error(f"Error saving paper technical terms to database: {e}")
        raise


def setup(fresh: bool = False, force: bool = False, chunk_size: int = 100):
    logger = setup_logger()
    if not fresh and erased_terms_file_path.is_file() and technical_terms_file_path.is_file():
        with open(erased_terms_file_path, "rb") as f:
            erased_terms = pickle.load(f)
        with open(technical_terms_file_path, "rb") as f:
            technical_terms = pickle.load(f)
    else:
        database_setup()
        erased_terms, technical_terms = generate_technical_terms(force, chunk_size)
        with open(erased_terms_file_path, "wb") as f:
            pickle.dump(erased_terms, f)
        with open(technical_terms_file_path, "wb") as f:
            pickle.dump(technical_terms, f)

        with open(atr_output_path.joinpath("technical_terms.txt"), "w") as f:
            for term, frequency in sorted(list(technical_terms.items()), key=lambda x: x[1], reverse=True):
                logger.info(f"{frequency}\t{term}")
        count_technical_terms_in_papers(force, chunk_size)
    return erased_terms, technical_terms


def define_subparser(subparser):
    atr = subparser.add_parser(subparser_name := 'atr')
    atr.add_argument('--fresh', dest='fresh', action='store_true',
                     help='Execute the computations no matter the results are already cached or not!')
    atr.add_argument('--force', dest='force', action='store_true',
                     help='Force truncate tables before inserting new data')
    atr.add_argument('--chunk-size', type=int, default=100,
                     help='Chunk size for parallel processing')
    return subparser_name

# if __name__ == '__main__':
#     setup(db_save=True, fresh=False)
