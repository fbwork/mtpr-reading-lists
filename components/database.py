import html
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional


from peewee import fn
from tqdm import tqdm

import utils
from models import (
    Paper, Author, PaperAuthor, PaperCitation, AuthorCitation, db
)
from utils import flag

current_path = Path(__file__).parent
db_output_path = current_path.joinpath('output', 'db')
db_papers: Dict[str, Paper] = {}
db_authors: Dict[str, Author] = {}


def setup_logger() -> logging.Logger:
    """Initialize logger with file output."""
    db_output_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=db_output_path.joinpath('database.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()


class QueryResult:
    """Class to store query results."""

    def __init__(self, query: str, papers: List[Paper], standard: Optional[str], ranked_paper_ids: List[int]):
        self.query = query
        self.papers = papers
        self.standard = standard
        self.ranked_paper_ids = ranked_paper_ids


def extract_authors(authors: str) -> Set[str]:
    """Extract authors from a semicolon-separated string."""
    return set(utils.split(html.unescape(authors), ';', strip_chars=' '))


def load_paper_text(paper_id: str) -> str | bool:
    """Load and clean paper text from file, with caching."""
    logger = setup_logger()
    cache_path = db_output_path.joinpath('paper_texts', f'{paper_id}.pkl')
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists() and not flag.is_module_done(f'load_paper_text_{paper_id}'):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached text for {paper_id}: {e}")

    if paper_id.startswith('L08'):
        logger.info(f"Skipping paper {paper_id}: Starts with 'L08'")
        return False

    file_path = current_path.parents[1].joinpath('ANN', '2014', 'papers_text', f'{paper_id}.txt')
    try:
        with open(file_path, encoding='utf-8', errors='ignore') as file:
            text = ""
            for line in file:
                text += line
                if len(text) > 1000:
                    break
    except (FileNotFoundError, IOError) as e:
        logger.warning(f"Cannot read file {file_path}: {e}")
        return False
    except UnicodeDecodeError as e:
        logger.error(f"Error decoding file {file_path}: {e}")
        return False

    if text.count(' the ') < 5 or len(text) < 100:
        logger.info(f"Skipping paper {paper_id}: Less than 5 'the' occurrences or less than 100 characters")
        return False

    clean_text = utils.remove(text, '\r\n ')
    if len(clean_text) < 100:
        logger.info(f"Skipping paper {paper_id}: Cleaned text too short ({len(clean_text)} characters)")
        return False

    text = utils.reg_remove(text, '\x00')
    text = utils.reg_replace(text, ['-\n', '\n', '\s+'], " ")
    text = html.unescape(text.lower())

    with open(cache_path, 'wb') as f:
        pickle.dump(text, f)
    flag.mark_module_done(f'load_paper_text_{paper_id}')
    return text


def is_data_corrupted(data: Dict) -> bool:
    """Check if metadata dictionary is corrupted."""
    required_keys = ['id', 'author', 'title', 'venue', 'year']
    return not all(key in data and data[key] for key in required_keys)


def extract_acl_metadata(file_path: Path, allow_ids: Optional[Set[str]] = None,
                         paper_body_needed: bool = False) -> Dict:
    """Extract metadata from ACL file, with caching."""
    logger = setup_logger()
    cache_path = db_output_path.joinpath('acl_metadata.pkl')

    if cache_path.exists() and not flag.is_module_done('extract_acl_metadata'):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached metadata: {e}")

    all_info = {}
    info = {}
    try:
        with open(file_path, encoding='utf-8', errors='ignore') as file:
            for line in tqdm(file.readlines(), desc=f"Parsing {os.sep.join(file_path.parts[-3:])}", ncols=100):
                meta = utils.split(line, '=', strip_chars='\n {}')
                if len(meta) < 2:
                    if info and not is_data_corrupted(info):
                        if paper_body_needed:
                            info['body'] = load_paper_text(info['id'])
                            if not info['body']:
                                info = {}
                                continue
                        if not allow_ids or info['id'] in allow_ids:
                            info['authors'] = extract_authors(info['author'])
                            del info['author']
                            all_info[info['id']] = info
                    info = {}
                else:
                    info[meta[0].strip()] = meta[1]
        if info and not is_data_corrupted(info):
            if paper_body_needed:
                info['body'] = load_paper_text(info['id'])
                if not info['body']:
                    return all_info
            if not allow_ids or info['id'] in allow_ids:
                info['authors'] = extract_authors(info['author'])
                del info['author']
                all_info[info['id']] = info
    except FileNotFoundError:
        logger.warning(f"Metadata file not found: {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing metadata file {file_path}: {e}")
        return {}

    with open(cache_path, 'wb') as f:
        pickle.dump(all_info, f)
    flag.mark_module_done('extract_acl_metadata')
    return all_info


def fill_paper_info():
    """Fill Paper, Author, and PaperAuthor tables, with resume support."""
    logger = setup_logger()
    if flag.is_module_done('fill_paper_info'):
        logger.info("Skipping fill_paper_info: Already completed")
        return db_papers, db_authors

    try:
        papers_2010 = extract_acl_metadata(current_path.parents[1].joinpath('ANN', '2010', 'acl-metadata.txt'))
        papers = extract_acl_metadata(
            current_path.parents[1].joinpath('ANN', '2014', 'acl-metadata.txt'),
            set(papers_2010.keys()),
            paper_body_needed=True
        )
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        raise

    new_papers = []
    new_authors = set()
    for i in tqdm(papers, desc="Preparing papers and authors...", ncols=100):
        try:
            new_papers.append(
                Paper(
                    code=i,
                    title=html.unescape(papers[i]['title']),
                    body=papers[i]['body'],
                    venue=papers[i]['venue'],
                    year=int(papers[i]['year'])
                )
            )
            new_authors.update(papers[i]['authors'])
        except (ValueError, KeyError) as e:
            logger.warning(f"Error processing paper {i}: {e}")
            continue

    new_authors = [Author(full_name=author) for author in new_authors]

    try:
        with db.atomic() as transaction:
            db.connect(reuse_if_open=True)
            Paper.truncate_table(cascade=True)
            Paper.bulk_create(new_papers, batch_size=400)
            Author.truncate_table(cascade=True)
            Author.bulk_create(new_authors, batch_size=400)
    except Exception as e:
        logger.error(f"Error saving papers or authors to database: {e}")
        transaction.rollback()
        raise
    finally:
        if not db.is_closed():
            db.close()

    db_papers.clear()
    db_papers.update({paper.code: paper for paper in Paper.select()})
    db_authors.clear()
    db_authors.update({author.full_name: author for author in Author.select()})

    new_paper_authors = []
    for code in tqdm(db_papers, desc="Preparing paper-author relations...", ncols=100):
        try:
            new_paper_authors.extend(
                PaperAuthor(paper=db_papers[code].id, author=db_authors[author].id)
                for author in papers[code]['authors'] if author in db_authors
            )
        except KeyError as e:
            logger.warning(f"Author not found for paper {code}: {e}")
            continue

    try:
        with db.atomic():
            db.connect(reuse_if_open=True)
            PaperAuthor.truncate_table(cascade=True)
            PaperAuthor.bulk_create(new_paper_authors, batch_size=400)
    except Exception as e:
        logger.error(f"Error saving paper-author relations to database: {e}")
        raise
    finally:
        if not db.is_closed():
            db.close()

    flag.mark_module_done('fill_paper_info')
    return db_papers, db_authors


def fill_paper_citations():
    """Fill PaperCitation table, with resume support."""
    logger = setup_logger()
    if flag.is_module_done('fill_paper_citations'):
        logger.info("Skipping fill_paper_citations: Already completed")
        return

    file_path = current_path.parents[1].joinpath('ANN', '2014', 'network', 'paper_citation_network.txt')
    try:
        with open(file_path, encoding='utf-8') as f:
            lines = list(f)
    except FileNotFoundError:
        logger.warning(f"Citation file not found: {file_path}")
        return
    except Exception as e:
        logger.error(f"Error reading citation file {file_path}: {e}")
        return

    new_paper_citations = []
    for line in tqdm(lines, desc="Inserting paper citations to DB...", ncols=100):
        try:
            paper_codes = utils.split(line, ['==>'], strip_chars=' \n')
            if len(paper_codes) != 2:
                logger.warning(f"Invalid citation format: {line.strip()}")
                continue
            if paper_codes[0] in db_papers and paper_codes[1] in db_papers:
                referring_id, cited_id = db_papers[paper_codes[0]], db_papers[paper_codes[1]]
                new_paper_citations.append(PaperCitation(citing_paper=referring_id, cited_paper=cited_id))
        except Exception as e:
            logger.warning(f"Error processing citation line {line.strip()}: {e}")
            continue

    try:
        with db.atomic():
            db.connect(reuse_if_open=True)
            PaperCitation.truncate_table(cascade=True)
            PaperCitation.bulk_create(new_paper_citations, batch_size=400)
    except Exception as e:
        logger.error(f"Error saving paper citations to database: {e}")
        raise
    finally:
        if not db.is_closed():
            db.close()

    flag.mark_module_done('fill_paper_citations')


def fill_author_citations():
    """Fill AuthorCitation table, with resume support."""
    logger = setup_logger()
    if flag.is_module_done('fill_author_citations'):
        logger.info("Skipping fill_author_citations: Already completed")
        return

    pc = PaperCitation.alias("pc")
    pa1, pa2 = PaperAuthor.alias('pa1'), PaperAuthor.alias('pa2')
    citing_author, cited_author = Author.alias('citing_author'), Author.alias('cited_author')

    citation_relations_query = (
        citing_author
        .select(citing_author.id, cited_author.id, fn.COUNT('*').alias('citations'))
        .join(pa1)
        .join(pc, on=(pc.citing_paper_id == pa1.paper))
        .join(pa2, on=(pa2.paper == pc.cited_paper))
        .join(cited_author, on=(pa2.author == cited_author.id))
        .group_by(citing_author.id, cited_author.id)
    )

    try:
        with db.atomic():
            db.connect(reuse_if_open=True)
            AuthorCitation.truncate_table(cascade=True)
            batch_size = 400
            citations = []
            for row in tqdm(citation_relations_query, desc="Computing author citations...", ncols=100):
                citations.append(
                    AuthorCitation(
                        citing_author=row.id,
                        cited_author=row.cited_author.id,
                        count=row.citations
                    )
                )
                if len(citations) >= batch_size:
                    AuthorCitation.bulk_create(citations, batch_size=batch_size)
                    citations = []
            if citations:
                AuthorCitation.bulk_create(citations, batch_size=batch_size)
    except Exception as e:
        logger.error(f"Error saving author citations to database: {e}")
        raise
    finally:
        if not db.is_closed():
            db.close()

    flag.mark_module_done('fill_author_citations')





def setup(fresh: bool = False, force: bool = False):
    logger = setup_logger()
    if not fresh and not force and flag.is_module_done('database_setup'):
        logger.info("Skipping database setup: Already completed")
        return

    tables = [
        Paper, Author, PaperAuthor, PaperCitation, AuthorCitation
    ]

    try:
        db.connect(reuse_if_open=True)
        for table in tables:
            if not table.table_exists():
                logger.info(f"Creating table {table.__name__}")
                db.create_tables([table])

            if force or fresh or not table.select().exists():
                logger.info(f"Populating table {table.__name__}...")
                if table in [Paper, Author, PaperAuthor]:
                    fill_paper_info()
                elif table == PaperCitation:
                    fill_paper_citations()
                elif table == AuthorCitation:
                    fill_author_citations()
                else:
                    logger.info(f"No fill function for {table.__name__} – skipping.")
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise
    finally:
        db.close()

    flag.mark_module_done('database_setup')


def define_subparser(subparser):
    """Define subparser for database command-line arguments."""
    database = subparser.add_parser(subparser_name := 'database')
    database.add_argument('--fresh', dest='fresh', action='store_true', help='Recompute even if cached.')
    database.add_argument('--force', dest='force', action='store_true', help='Force truncate tables.')
    return subparser_name


def get_relevant_papers(query: str):
    """Retrieve relevant paper IDs with caching."""
    logger = setup_logger()
    query = query.lower()
    cache_path = db_output_path.joinpath('query_cache', f'{hash(query)}.pkl')
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists() and not flag.is_module_done(f'query_{hash(query)}'):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached query results for '{query}': {e}")

    try:
        with db.atomic():
            db.connect(reuse_if_open=True)
            papers = Paper.select().where(
                fn.LOWER(Paper.title).contains(query) | fn.LOWER(Paper.body).contains(query)
            ).limit(1000)
            result = [p.id - 1 for p in papers]
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            flag.mark_module_done(f'query_{hash(query)}')
            return result
    except Exception as e:
        logger.error(f"Error querying relevant papers for '{query}': {e}")
        return []
    finally:
        if not db.is_closed():
            db.close()
