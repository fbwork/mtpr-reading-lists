import os
import pickle
from pathlib import Path
from threading import Lock

import pandas as pd
from dotenv import load_dotenv
from peewee import *
from playhouse.signals import Model as PModel

load_dotenv(Path(__file__).parent.parent.joinpath('.env'))

db = PostgresqlDatabase(os.getenv('POSTGRES_DB'), user=os.getenv('POSTGRES_USER'),
                        password=os.getenv('POSTGRES_PASSWORD'), host=os.getenv('POSTGRES_HOST'),
                        port=os.getenv('POSTGRES_PORT'))

ThroughDeferred1 = DeferredThroughModel()

DEFAULT_PER_PAGE = 25


class BaseModel(PModel):
    class Meta:
        database = db

    def refresh(self):
        return type(self).get(self._pk_expr())


class Author(BaseModel):
    id = AutoField()
    full_name = CharField(max_length=100)

    @property
    def formatted_name(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def first_name(self):
        return str(self.full_name).split(',')[1].strip()

    @property
    def last_name(self):
        return str(self.full_name).split(',')[0].strip()

    @property
    def total_citations(self):
        return (Paper
                .select(fn.COUNT(Paper.id))
                .join(PaperCitation, on=(PaperCitation.citing_paper_id == Paper.id))
                .where(PaperCitation.cited_paper_id.in_(self.papers.select(Paper.id)))
                .scalar()
                )

    def outcited_papers(self, per_page=100, page=1, count=False):
        if count:
            return (Paper
                    .select(fn.COUNT(Paper.id))
                    .join(PaperCitation, on=(PaperCitation.citing_paper_id == Paper.id))
                    .where(PaperCitation.citing_paper_id.in_(self.papers.select(Paper.id)))
                    .scalar()
                    )

        else:
            return (Paper
                    .select(Paper)
                    .join(PaperCitation, on=(PaperCitation.citing_paper_id == Paper.id))
                    .where(PaperCitation.citing_paper_id.in_(self.papers.select(Paper.id)))
                    .order_by(Paper.year.desc())
                    .paginate(page, per_page)
                    )

    def incited_papers(self, per_page=100, page=1, count=False):
        if count:
            return (Paper
                    .select(fn.COUNT(PaperCitation.citing_paper_id))
                    .join(PaperCitation, on=(PaperCitation.cited_paper_id == Paper.id))
                    .where(PaperCitation.cited_paper_id.in_(self.papers.select(Paper.id)))
                    .scalar()
                    )

        else:
            return (Paper
                    .select(Paper, fn.COUNT(PaperCitation.citing_paper_id).alias('citations'))
                    .join(PaperCitation, on=(PaperCitation.cited_paper_id == Paper.id))
                    .where(PaperCitation.cited_paper_id.in_(self.papers.select(Paper.id)))
                    .group_by(Paper)
                    .order_by(SQL('citations').desc())
                    .paginate(page, per_page)
                    )

    def incited_authors(self, per_page=None, page=1, count=False):
        if count:
            return (Author
                    .select(fn.COUNT(Author.full_name))
                    .join(PaperAuthor)
                    .join(PaperCitation, on=(PaperCitation.citing_paper_id == PaperAuthor.paper))
                    .where(PaperCitation.cited_paper_id.in_(self.papers.select(Paper.id)))
                    .scalar()
                    )

        else:
            q = (Author
                 .select(Author, fn.COUNT(Author.full_name).alias('citations'))
                 .distinct()
                 .join(PaperAuthor)
                 .join(PaperCitation, on=(PaperCitation.citing_paper_id == PaperAuthor.paper))
                 .where(PaperCitation.cited_paper_id.in_(self.papers.select(Paper.id)))
                 .group_by(Author)
                 .order_by(SQL('citations').desc())
                 )
            if per_page:
                q = q.paginate(page, per_page)

            return q

    def outcited_authors(self, per_page=None, page=1, count=False):
        if count:
            return (Author
                    .select(fn.COUNT(Author.full_name))
                    .join(PaperAuthor)
                    .join(PaperCitation, on=(PaperCitation.cited_paper_id == PaperAuthor.paper))
                    .where(PaperCitation.citing_paper_id.in_(self.papers.select(Paper.id)))
                    .scalar()
                    )

        else:
            q = (Author
                 .select(Author, fn.COUNT(Author.full_name).alias('citations'))
                 .distinct()
                 .join(PaperAuthor)
                 .join(PaperCitation, on=(PaperCitation.cited_paper_id == PaperAuthor.paper))
                 .where(PaperCitation.citing_paper_id.in_(self.papers.select(Paper.id)))
                 .group_by(Author)
                 .order_by(SQL('citations').desc())
                 )
            if per_page:
                q = q.paginate(page, per_page)

            return q

    def self_cited_papers(self, per_page=100, page=1, count=False):
        if count:
            return (PaperCitation
                    .select(fn.COUNT(PaperCitation.cited_paper_id))
                    .join(PaperAuthor, on=(PaperCitation.citing_paper_id == PaperAuthor.paper))
                    .where(PaperAuthor.author == self.full_name,
                           PaperCitation.cited_paper_id.in_(self.papers.select(Paper.id)))
                    .scalar()
                    )

        else:
            return (Paper
                    .select(Paper)
                    .join(PaperCitation, on=(PaperCitation.cited_paper_id == Paper.id))
                    .join(PaperAuthor, on=(PaperCitation.citing_paper_id == PaperAuthor.paper))
                    .where(PaperAuthor.author == self.full_name,
                           PaperCitation.cited_paper_id.in_(self.papers.select(Paper.id)))
                    .paginate(page, per_page)
                    )

    @staticmethod
    def document_authors(doc_ids: list):
        return (Author
                .select(Author)
                .distinct(Author)
                .join(PaperAuthor)
                .where(PaperAuthor.paper << doc_ids))

    @staticmethod
    def most_cited(per_page=100, page=1):
        return (Author
                .select(Author, fn.COUNT(PaperCitation.citing_paper_id).alias('citations'))
                .join(PaperAuthor, on=(PaperAuthor.author == Author.id))
                .join(PaperCitation, on=(PaperCitation.cited_paper_id == PaperAuthor.paper))
                .group_by(Author)
                .order_by(SQL('citations').desc())
                .paginate(page, per_page)
                )

    @staticmethod
    def most_referencing(per_page=100, page=1):
        return (Author
                .select(Author, fn.COUNT(PaperCitation.cited_paper_id).alias('referencing'))
                .join(PaperAuthor, on=(PaperAuthor.author == Author.id))
                .join(PaperCitation, on=(PaperCitation.citing_paper_id == PaperAuthor.paper))
                .group_by(Author)
                .order_by(SQL('referencing').desc())
                .paginate(page, per_page)
                )

    @staticmethod
    def most_self_referencing(per_page=100, page=1):
        return (Author
                .alias('citing')
                .select(SQL('citing.full_name'), fn.COUNT(SQL('pc.citing_paper_id')).alias('referencing'))
                .join(PaperAuthor.alias('pa1'), on=(PaperAuthor.alias('pa1').author == Author.alias('citing').id))
                .join(PaperCitation.alias('pc'),
                      on=(PaperCitation.alias('pc').cited_paper_id == PaperAuthor.alias('pa1').paper))
                .join(PaperAuthor.alias('pa2'),
                      on=(PaperAuthor.alias('pa2').paper == PaperCitation.alias('pc').citing_paper_id))
                .join(Author.alias('cited'), on=(Author.alias('cited').id == PaperAuthor.alias('pa2').author))
                .where(SQL('citing.full_name = cited.full_name'))
                .group_by(SQL('citing.full_name'))
                .order_by(SQL('referencing').desc())
                .paginate(page, per_page)
                )

    @staticmethod
    def most_active(per_page=100, page=1):
        return (Author
                .select(Author, fn.COUNT(PaperAuthor.paper).alias('no_papers'))
                .join(PaperAuthor)
                .group_by(Author)
                .order_by(SQL('no_papers').desc())
                .paginate(page, per_page)
                )

    def __hash__(self):
        return hash(self.full_name)

    class Meta:
        db_table = 'authors'


class AuthorCitation(BaseModel):
    citing_author = ForeignKeyField(Author, on_delete='Cascade', on_update='Cascade')  # Citing
    cited_author = ForeignKeyField(Author, on_delete='Cascade', on_update='Cascade')  # Cited
    count = IntegerField(default=1)

    class Meta:
        primary_key = CompositeKey('citing_author', 'cited_author')
        db_table = 'author_citations'


class Paper(Model):
    id = AutoField()
    code = CharField(unique=True)
    title = TextField()
    body = TextField(null=True)
    venue = CharField()
    year = IntegerField()
    authors = ManyToManyField(Author, backref='papers', through_model=ThroughDeferred1)

    refs = {}
    refs_lock = Lock()

    class Meta:
        database = db

    @classmethod
    def paper_authors(cls, paper_id):
        return Author.select().join(PaperAuthor).where(PaperAuthor.paper == paper_id)

    @staticmethod
    def code2id(code):
        if type(code) is not list:
            code = [code]

        papers = sorted(list(Paper.select().where(Paper.code << code)), key=lambda x: code.index(x.code))
        return [paper.id for paper in papers]

    def terms(self):
        return TechnicalTerm.select(TechnicalTerm).join(PaperTechnicalTerm).where(PaperTechnicalTerm.paper == self.id)

    def term_frequency(self, term: str):
        t = PaperTechnicalTerm.select(PaperTechnicalTerm, TechnicalTerm).join(TechnicalTerm).where(
            (TechnicalTerm.text == term) & (PaperTechnicalTerm.paper == self.id))

        if t.exists():
            return t.get().frequency
        else:
            return 0

    @staticmethod
    def paper_references(paper_id):
        if Paper.refs:
            return Paper.refs[paper_id]

        else:
            return (Paper
                    .select(Paper)
                    .join(PaperCitation, on=(PaperCitation.cited_paper_id == Paper.id))
                    .where(PaperCitation.citing_paper_id == paper_id))

    @staticmethod
    def paper_authors(paper_id):
        return (Author
                .select(Author)
                .join(PaperAuthor)
                .where(PaperAuthor.paper == paper_id))

    def references(self, per_page=None, page=1, count=False):
        if count:
            return (Paper
                    .select(fn.COUNT(Paper.id))
                    .join(PaperCitation, on=(PaperCitation.cited_paper_id == Paper.id))
                    .where(PaperCitation.citing_paper_id == self.id)
                    .scalar()
                    )

        else:
            q = (Paper
                 .select(Paper)
                 .join(PaperCitation, on=(PaperCitation.cited_paper_id == Paper.id))
                 .where(PaperCitation.citing_paper_id == self.id))

            if per_page:
                q = q.paginate(page, per_page)

            return q

    def citations(self, per_page=None, page=1, count=False):
        if count:
            return (Paper
                    .select(fn.COUNT(Paper.id))
                    .join(PaperCitation, on=(PaperCitation.citing_paper_id == Paper.id))
                    .where(PaperCitation.cited_paper_id == self.id)
                    .scalar())

        else:
            q = (Paper
                 .select(Paper)
                 .join(PaperCitation, on=(PaperCitation.citing_paper_id == Paper.id))
                 .where(PaperCitation.cited_paper_id == self.id))

        if per_page:
            q = q.paginate(page, per_page)

        return q

    @staticmethod
    def most_cited(per_page=100, page=1):
        return (Paper
                .select(Paper.id, fn.COUNT(PaperCitation.citing_paper_id).alias('citations'))
                .join(PaperCitation, on=(PaperCitation.cited_paper_id == Paper.id))
                .group_by(Paper.id)
                .order_by(SQL('citations').desc())
                .paginate(page, per_page))

    @classmethod
    def most_referencing(cls, per_page=DEFAULT_PER_PAGE, page=1):
        return (Paper
                .select(Paper.id, fn.COUNT(PaperCitation.cited_paper_id).alias('references'))
                .join(PaperCitation, on=(PaperCitation.citing_paper_id == Paper.id))
                .group_by(Paper.id)
                .order_by(SQL('references').desc())
                .paginate(page, per_page))

    def technical_terms(self, per_page=DEFAULT_PER_PAGE, page=1, count=False):
        query = (PaperTechnicalTerm
                 .select(PaperTechnicalTerm, Paper, TechnicalTerm)
                 .join(Paper)
                 .switch(PaperTechnicalTerm)
                 .join(TechnicalTerm)
                 .where(PaperTechnicalTerm.paper == self.id))

        if count:
            return query.count()
        elif not page:
            return query
        else:
            return query.paginate(page, per_page)

    @staticmethod
    def dump_titles():
        papers = [{"id": p.id, "title": p.title} for p in list(Paper.select(Paper.id, Paper.title))]
        papers_df = pd.DataFrame(papers)
        print(papers_df.head())
        papers_df.to_csv('output/brt/paper_title.csv')


class PaperCitation(BaseModel):
    citing_paper = ForeignKeyField(Paper, null=False, on_delete='Cascade')
    cited_paper = ForeignKeyField(Paper, null=False, on_delete='Cascade')

    class Meta:
        primary_key = CompositeKey('citing_paper', 'cited_paper')
        db_table = 'paper_citations'


class PaperAuthor(BaseModel):
    paper = ForeignKeyField(Paper)
    author = ForeignKeyField(Author)

    class Meta:
        primary_key = CompositeKey('paper', 'author')
        db_table = 'paper_authors'


# class TitleTopic(BaseModel):
#     topic_id = IntegerField(primary_key=True)
#     name = TextField()
#     count = IntegerField()
#
#     class Meta:
#         db_table = 'title_topics'
#
#
# class BodyTopic(BaseModel):
#     topic_id = IntegerField(primary_key=True)
#     name = TextField()
#     count = IntegerField()
#
#     class Meta:
#         db_table = 'body_topics'


# class PaperTitleTopic(BaseModel):
#     paper = ForeignKeyField(Paper, backref='title_topics')
#     topic = ForeignKeyField(TitleTopic, field='topic_id', backref='papers')
#     probability = FloatField()
#
#     class Meta:
#         db_table = 'paper_title_topics'
#         indexes = ((('paper', 'topic'), True),)
#
#
# class PaperBodyTopic(BaseModel):
#     paper = ForeignKeyField(Paper, backref='body_topics')
#     topic = ForeignKeyField(BodyTopic, backref='body_papers')
#     probability = FloatField()
#
#     class Meta:
#         db_table = 'paper_body_topics'
#         primary_key = CompositeKey('paper', 'topic')


ThroughDeferred1.set_model(PaperAuthor)


class TechnicalTerm(BaseModel):
    id = AutoField(primary_key=True)
    text = CharField()
    frequency = IntegerField()

    class Meta:
        db_table = 'technical_terms'

    def papers(self, per_page=DEFAULT_PER_PAGE, page=1, count=False):
        query = (PaperTechnicalTerm
                 .select(PaperTechnicalTerm, Paper, TechnicalTerm)
                 .join(Paper)
                 .switch(PaperTechnicalTerm)
                 .join(TechnicalTerm)
                 .where(PaperTechnicalTerm.technical_term == self.id))

        if count:
            return query.count()
        else:
            return query.paginate(page, per_page)


class PaperTechnicalTerm(BaseModel):
    paper = ForeignKeyField(Paper, null=False, on_delete='Cascade')
    technical_term = ForeignKeyField(TechnicalTerm, null=False, on_delete='Cascade')
    frequency = IntegerField(default=0)

    class Meta:
        primary_key = CompositeKey('paper', 'technical_term')
        db_table = 'paper_technical_terms'

    @staticmethod
    def verbose():
        return (PaperTechnicalTerm
                .select(TechnicalTerm, Paper, PaperTechnicalTerm)
                .join(TechnicalTerm)
                .switch(PaperTechnicalTerm)
                .join(Paper))


def create_tables():
    with db:
        db.create_tables([Author, AuthorCitation, Paper, PaperTechnicalTerm, PaperCitation, PaperAuthor, TechnicalTerm])


def setup_paper_refs():
    refs_file = Path(__file__).parent.joinpath('refs.bin')
    with Paper.refs_lock:
        if refs_file.exists() and not Paper.refs:
            try:
                with refs_file.open('rb') as f:
                    Paper.refs = pickle.load(f)
                print('Reference caches were found!')
            except (pickle.PickleError, EOFError) as e:
                print(f"Error loading reference cache: {e}")
                Paper.refs = {}
        else:
            print("Creating reference caches...")
            tmp = {}
            try:
                for paper_id in range(1, Paper.select().count() + 1):
                    tmp[paper_id] = [p.id for p in Paper.paper_references(paper_id)]
                with refs_file.open('wb') as f:
                    pickle.dump(tmp, f)
                Paper.refs = tmp
                print("Reference caches created!")
            except Exception as e:
                print(f"Error creating reference cache: {e}")
                raise
