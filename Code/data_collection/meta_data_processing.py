import pymongo
from pydriller import Repository
import json
import datetime
from loguru import logger
import argparse
import time
import calendar
import nltk
import pandas as pd
import ast
from github import Github, GithubException, RateLimitExceededException
from credential import ACCESS_TOKENS


from main import fetch_file, delete_file
from filters import branch_filtered, filter_message
from configurations import KEYWORDS, initialize_configuration

client = pymongo.MongoClient("mongodb://root:example@localhost:27017/")
db = client.meta_data
raw_meta_data = db.raw_meta_data


lemmatiser = nltk.WordNetLemmatizer()
tokenizer = nltk.SpaceTokenizer()

def patch_helper(date, hour, logger):
    i = 0

    with open(date + "-" + str(hour) + ".json") as f:
        for line in f.readlines():
            try:
                i += 1

                instance = json.loads(line.strip())

                if instance['type'] == "PushEvent":
                    instance_payload = instance['payload']
                    commit_message_list, commit_sha_list = filter_message(instance_payload)
                    if len(commit_message_list) != 0:
                        if not branch_filtered(instance):
                            record = raw_meta_data.find_one({'repo_name': instance['repo']['name'].strip()})
                            if record:
                                create_time = record['created_at']
                                if datetime.datetime.strptime(create_time, "%Y-%m-%dT%H:%M:%SZ") < date.replace(hour=hour):
                                    raw_meta_data.delete_one({'repo_name': instance['repo']['name'].strip()})
                                    raw_meta_data.insert_one({'repo_name': instance['repo']['name'].strip(),
                                                                'commit_id': commit_sha_list,
                                                                'commit_message': commit_message_list, 
                                                                'commit_date': instance['created_at']})                          
                continue

            except Exception as e:
                logger.info(f"{i}th record has issue: repo name {instance['repo']['name']}")
                continue


def patch_database(end_date: datetime.datetime, start_date: datetime.datetime, end_hour: int, start_hour: int, logger):
    """
    During the first run, there will be some left json files that we missed. Add it to database by a second run on those time frames.
    """
    days = (end_date - start_date).days
    for i in range(days+1):
        date = (end_date - datetime.timedelta(i)).strftime("%Y-%m-%d")
        # breakpoint()

        if date == end_date.strftime("%Y-%m-%d"):
            for hour in reversed(range(end_hour)):
                fetch_file(date, hour)
                patch_helper(date, hour, logger)
                delete_file(date, hour)
        elif date == start_date.strftime("%Y-%m-%d"):
            for hour in reversed(range(start_hour, 24)):
                fetch_file(date, hour)
                patch_helper(date, hour, logger)
                delete_file(date, hour)
        else:
            for hour in reversed(range(24)):
                fetch_file(date, hour)
                patch_helper(date, hour, logger)
                delete_file(date, hour)
    
    logger.info("Patch database finished.")



def filter_1(db, collection):
    """
    Filter out repositories that are forked, or does not have pull requests.
    Transfer all the valid repositories to filter_1 collection.
    The indexed field in filter_1 collection is 'repo_name'
    """
    raw_meta_data = db[collection]
    db.filter_1.create_index('repo_name')
    for repo_record in raw_meta_data.find({}):
        if repo_record['is_fork'] == False and repo_record['PR'] == True:
            db.filter_1.insert_one(repo_record)
    
def filter_2(db, g, logger, collection='filter_1', id=0):
    """
    First go through all records in filter_1 collection, find the repositories that does not have language field,
    use Github API to get the language of the repository, and update the record in filter_1 collection.

    Then, go through all records in filter_1 collection, find the repositories that are written in Python or Java,
    use Github API to get the commit message that changes README file, apply the keyword filter, and insert the record
    into filter_2 collection.

    There could be too many records, specify the id field to use different Github API keys.
    """
    filter_1 = db[collection]
    length = filter_1.count_documents({})
    idx = 0 

    # update language field
    while idx < length:
        try:
            cursor = filter_1.find({}, no_cursor_timeout=True).sort('repo_name', pymongo.ASCENDING)
            cursor.skip(idx)
            for repo_record in cursor:
                try:
                    # comment this line out when finalising the code, this line is just for fast recovery.
                    if idx % 1000 == 0:
                        core_rate_limit = g.get_rate_limit().core
                        if core_rate_limit.remaining < 20:
                            reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
                            sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 15
                            logger.info(f"current scanned {idx} reords")
                            logger.info(f"rate limit lower than 20, sleep for {sleep_time} seconds")
                            time.sleep(sleep_time)
                    if idx % len(ACCESS_TOKENS) == id:
                        if repo_record['language'] == None:
                            language = g.get_repo(repo_record['repo_name']).language
                            if language is not None:
                                filter_1.update_one({'repo_name': repo_record['repo_name']}, {"$set": {"language": language}})
                except RateLimitExceededException as e:
                    core_rate_limit = g.get_rate_limit().core
                    reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
                    sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 15
                    logger.info(f"exceeding rate limit, sleep for {sleep_time} seconds")
                    time.sleep(sleep_time)
                except GithubException as e:
                    # if fall in heccessible, it cre, the repository is not aould happen either because the repository is deleted, renamed, or the repository is private.
                    pass
                except Exception as e:
                    logger.info(f"unknown exception: {e} at {idx}th record")
                
                idx += 1
        except Exception as e:
            logger.info(f"expired cursor at {idx} record")
            continue

    
    logger.info("language field update finished")

    filter_2 = db['filter_2']
    invalid_repos = db['invalid_repos']
    # get commit number
    idx = 0
    while idx < length:
        try:
            cursor = filter_1.find({}, no_cursor_timeout=True).sort('repo_name', pymongo.ASCENDING)
            cursor.skip(idx)
            for repo_record in cursor:
                try:
                    if idx % 1000 == 0:
                        core_rate_limit = g.get_rate_limit().core
                        if core_rate_limit.remaining < 20:
                            reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
                            sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 15
                            logger.info(f"current scanned {idx} reords")
                            logger.info(f"rate limit lower than 20, sleep for {sleep_time} seconds")
                            time.sleep(sleep_time)
                    if idx % len(ACCESS_TOKENS) == id:
                        if repo_record['language'] == 'Python' or repo_record['language'] == 'Java':
                            repo = g.get_repo(repo_record['repo_name'])
                            star_count = repo.stargazers_count
                            commit_count = repo.get_commits().totalCount
                            filter_2.insert_one({'repo_name': repo_record['repo_name'], 'commit_count': commit_count, 'star_count': star_count, 'language': repo_record['language']})
                except RateLimitExceededException as e:
                    core_rate_limit = g.get_rate_limit().core
                    reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
                    sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 15
                    logger.info(f"current scanned {idx} reords")
                    logger.info(f"exceeding rate limit, sleep for {sleep_time} seconds")
                    time.sleep(sleep_time)
                except GithubException as e:
                    # if fall in here, the repository is not accessible, it could happen either because the repository is deleted, renamed, or the repository is private.
                    # we store all them, so we could report the number of repositories that are not accessible in the paper.
                    invalid_repos.insert_one({'repo_name': repo_record['repo_name'], 'language': repo_record['language']})
                except Exception as e:
                    logger.info(f"unknown exception: {e} in {repo_record['repo_name']}")
                idx += 1
        except Exception as e:
            logger.info(f"expired cursor at {idx} record")
            continue
    logger.info("filter_2 finished")


def remove_forks(g, id):
    filter_2 = db['filter_2']
    filter_3 = db['filter_3']
    length = filter_2.count_documents({})
    # get commit number
    idx = 0
    while idx < length:
        try:
            cursor = filter_2.find({}, no_cursor_timeout=True).sort('repo_name', pymongo.ASCENDING)
            cursor.skip(idx)
            for repo_record in cursor:
                try:
                    if idx % 2000 == 0:
                        core_rate_limit = g.get_rate_limit().core
                        if core_rate_limit.remaining < 20:
                            reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
                            sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 15
                            logger.info(f"current scanned {idx} reords")
                            logger.info(f"rate limit lower than 20, sleep for {sleep_time} seconds")
                            time.sleep(sleep_time)

                    if idx % len(ACCESS_TOKENS) == id:
                        if repo_record['commit_count'] >= 500 and repo_record['star_count'] >= 10:
                            repo = g.get_repo(repo_record['repo_name'])
                            if repo.fork == False:
                                filter_3.insert_one(repo_record)
                except RateLimitExceededException as e:
                    core_rate_limit = g.get_rate_limit().core
                    reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
                    sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 15
                    logger.info(f"current scanned {idx} reords")
                    logger.info(f"exceeding rate limit, sleep for {sleep_time} seconds")
                    time.sleep(sleep_time)
                except GithubException as e:
                    # if fall in here, the repository is not accessible, it could happen either because the repository is deleted, renamed, or the repository is private.
                    # we store all them, so we could report the number of repositories that are not accessible in the paper.
                    continue
                except Exception as e:
                    logger.info(f"unknown exception: {e} in {repo_record['repo_name']}")
                idx += 1
        except Exception as e:
            logger.info(f"expired cursor at {idx} record")
            continue
    logger.info("filter_2 finished")


def modify_repo_name(g, id):
    filter_3 = db['filter_3']
    length = filter_3.count_documents({})
    # get commit number
    idx = 0
    while idx < length:
        try:
            cursor = filter_3.find({}, no_cursor_timeout=True).sort('repo_name', pymongo.ASCENDING)
            cursor.skip(idx)
            for repo_record in cursor:
                try:
                    if idx % 2000 == 0:
                        core_rate_limit = g.get_rate_limit().core
                        if core_rate_limit.remaining < 20:
                            reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
                            sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 15
                            logger.info(f"current scanned {idx} reords")
                            logger.info(f"rate limit lower than 20, sleep for {sleep_time} seconds")
                            time.sleep(sleep_time)

                    if idx % len(ACCESS_TOKENS) == id:
                        if g.get_repo(repo_record['repo_name']).full_name != repo_record['repo_name']:
                            filter_3.update_one({'repo_name': repo_record['repo_name']}, {'$set': {'repo_name': g.get_repo(repo_record['repo_name']).full_name}})
                except RateLimitExceededException as e:
                    core_rate_limit = g.get_rate_limit().core
                    reset_timestamp = calendar.timegm(core_rate_limit.reset.timetuple())
                    sleep_time = reset_timestamp - calendar.timegm(time.gmtime()) + 15
                    logger.info(f"current scanned {idx} reords")
                    logger.info(f"exceeding rate limit, sleep for {sleep_time} seconds")
                    time.sleep(sleep_time)
                except GithubException as e:
                    # if fall in here, the repository is not accessible, it could happen either because the repository is deleted, renamed, or the repository is private.
                    # we store all them, so we could report the number of repositories that are not accessible in the paper.
                    continue
                except Exception as e:
                    logger.info(f"unknown exception: {e} in {repo_record['repo_name']}")
                idx += 1
        except Exception as e:
            logger.info(f"expired cursor at {idx} record")
            continue
    logger.info("filter_2 finished")





def filter_3(db, logger):
    """
    This filter will use pydriller to see the repository commits history. 
    It will only keep repositories with commits that modifiy the README file and at the same time, have commit message that contains our predefined keywords.
    """
    filter_3 = db['filter_3']
    filter_4 = db['filter_4']

    idx = 0
    length = filter_3.count_documents({})
    while idx < length:
        try: 
            cursor = filter_3.find({}).sort('repo_name', pymongo.ASCENDING)
            cursor.skip(idx)
            for repo_record in cursor:
                # discuss this part in the meeting. Do we only look at repositories with more than 500 commits or do we look at repositories with more than 500 commits + more than 10 stars?
                # Also, we need to discuss whether this 500 commits will all on the main branch, or the commit sum of all branches.
                # Last, we need to discuss the threshold of the number of commits that modify the README file. This is necessary because we want to see the how README files change over time.
                # Could also just collect all of them and decide after the statistics. But I would argue at least 3-4 commits that modify README files.
                # As of now, just implement it, and I will change it later after the discussion.

                upper_limit = 0
                idx += 1
                if idx % 20 == 0:
                    logger.info(f"current scanned {idx} reords")
                if repo_record['language'] == 'Java':
                    upper_limit = 5099
                elif repo_record['language'] == 'Python':
                    upper_limit = 3690
                try:
                    if 500 <= repo_record['commit_count'] <= upper_limit and repo_record['star_count'] >= 10:
                        ### set a upper limit for the commit count.
                        README_shas = []  # we can infer the number of readme commits, and readme shas will be used in the visualization phase.
                        flag = False
                        repo_url = f"https://github.com/{repo_record['repo_name']}.git"

                        for commit in Repository(repo_url, filepath="README.md").traverse_commits():
                            commit_message = commit.msg
                            
                            README_shas.append(commit.hash)
                            if not flag and filter_message(commit_message):
                                flag = True
                        if flag:
                            filter_4.insert_one({'repo_name': repo_record['repo_name'], 'commit_counts': repo_record['commit_count'], 'language': repo_record['language'], 'README_shas': README_shas})
                except Exception as e:
                    logger.info(e)
                    logger.info(f"repo {repo_record['repo_name']} has issues")
        except Exception as e:
            logger.info(f"expired cursor at {idx} record")
            continue
    logger.info("filter_3 finished")

def filter_message(message):
    
    message = message.lower().strip()

    if message.endswith('.'):
        message = message.rstrip('.')
    message_without_punct = message.replace(',', '').replace('!', '').replace('?', '')

    message_tokens = tokenizer.tokenize(message_without_punct)
    print(message_tokens)
    for token in message_tokens:
        token = token.lower().strip().rstrip('.')
        lemma = lemmatiser.lemmatize(token, "v")
        # lemmatise the word and compare with our keyword set.
        if lemma == token:
            lemma = lemmatiser.lemmatize(token, "n")
            if lemma == token:
                lemma = lemmatiser.lemmatize(token, "a")
    
        if lemma in KEYWORDS:
            return True
  
    return False
    




def update_records(db, collection):
    """
    update all record in the collection with extra attributes, including 'PR', 'language', 'is_fork'. all default to None
    """
    my_collection = db[collection]
    my_collection.update_many({}, {"$set": {"PR": False, "language": None, "is_fork": False}})

        


def meta_data_addition(date, hour, logger):
    client = pymongo.MongoClient("mongodb://root:example@localhost:27017/")
    db = client.meta_data
    raw_meta_data = db.raw_meta_data

    i = 0
    with open(date + "-" + hour + ".json") as f:
        for line in f.readlines():
            try:
                i += 1
                
                instance = json.loads(line.strip())
                if instance['type'] == 'ForkEvent':
                    original = instance['repo']['name']
                    fork = instance['payload']['forkee']['full_name']
                    # check wheher this repo name has already appeared in the database
                    raw_meta_data.update_one({'repo_name': fork}, {"$set": {"is_fork": True}})
                 
                elif instance['type'] == 'PullRequestEvent':
                    try:
                        head = instance['payload']['pull_request']['head']['repo']['full_name']
                    except:
                        head = None
                    base = instance['payload']['pull_request']['base']['repo']['full_name']
                    language = instance['payload']['pull_request']['base']['repo']['language']
                    
                    base_db_entry = raw_meta_data.find_one({'repo_name': base})
                    if base_db_entry and base_db_entry['PR'] == False:
                        raw_meta_data.update_one({'repo_name': base}, {"$set": {"PR": True, "language": language}})

                    if head is not None:
                        head_db_entry = raw_meta_data.find_one({'repo_name': head})
                        if head_db_entry and head_db_entry['language'] is None:
                            raw_meta_data.update_one({'repo_name': head}, {"$set": {"language": language}})

            except Exception as e:
                logger.info(f"{i}th record has issue: repo name {instance['repo']['name']}")
                logger.info(e)
                continue



def sampled_data_addition():
    data = pd.read_csv('sampled_data.csv')
   

    data['README_shas'] = data['README_shas'].apply(ast.literal_eval)
    for i in range(len(data)):
        repo_name = data.loc[i, 'repo_name']
        repo_url = f"https://github.com/{repo_name}.git"
        for commit in Repository(repo_url, filepath='README.md').traverse_commits():
            commit_message = commit.msg
            print(commit_message)
            for file in commit.modified_files:
                if file.filename == "README.md":
                    content = file.content.decode('utf-8')
                    print(content)
                    print("==========")


if __name__ == "__main__":

    sampled_data_addition()











    # filter_1(db, 'raw_meta_data')

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--id", type=int)
    # args = parser.parse_args()

    # g = initialize_configuration(args.id)
    # logger.add(f"commit_filter.log")

    # filter_3(db, logger)   


    # remove_forks(g, id=args.id)
    # update_records(db, 'raw_meta_data')


    # patch the database
    # patch_database(datetime.datetime(2015, 1, 23), datetime.datetime(2015, 1, 23), 21, 0, logger)

    # start_date = datetime.date(2023, 2, 28)
    # end_date = datetime.date(2015, 1, 1)
    # days = (start_date - end_date).days
  
    # for day in range(days+1):
    #     if day % 7 != args.id:
    #         continue
    #     date = (start_date - datetime.timedelta(day)).strftime('%Y-%m-%d')
    #     for hour in reversed(range(24)):
    #         try:
    #             fetch_file(date, hour)
    #             logger.info(f"extracting data from {date} - {hour}")
    #             meta_data_addition(date, str(hour), logger)
    #             delete_file(date, hour)
    #         except:
    #             logger.info(f"failed to extract data from {date} - {hour}")
    #             continue


    #  client = pymongo.MongoClient("mongodb://root:example@localhost:27017/")
    #  print(list(client.meta_data.pull_requests.find({})))

   