from credential import ACCESS_TOKENS
from github import Github
import pymongo

TARGET_FILES = set()
TARGET_LANGUAGES = set()
KEYWORDS = set()



def initialize_configuration(id):
    with open('target_file.txt') as f:
        for line in f.readlines():
            TARGET_FILES.add(line.strip())

    with open('investigated_languages.txt') as f:
        for line in f.readlines():
            TARGET_LANGUAGES.add(line.strip())

    with open("keywords.txt") as f:
        for line in f.readlines():
            KEYWORDS.add(line.strip())

    client = pymongo.MongoClient("mongodb://root:example@localhost:27017/")
    db = client.meta_data
    my_collection = db.raw_meta_data
    my_collection.create_index([('repo_name', pymongo.ASCENDING)])

    g = Github(ACCESS_TOKENS[id])

    return g
