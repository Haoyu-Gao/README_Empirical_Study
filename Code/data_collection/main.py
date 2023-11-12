import json

from loguru import logger
import subprocess
import calendar
import pickle
import base64
import argparse

import github
from github import Github, RateLimitExceededException, UnknownObjectException
import csv
import time
import datetime
import pymongo


from credential import *
from configurations import initialize_configuration


from filters import star_language_filtered, branch_filtered, filter_message, filter_file





def meta_data_extraction(date, hour, g, logger):
    i = 0
    client = pymongo.MongoClient("mongodb://root:example@localhost:27017/")
    db = client.meta_data
    my_collection = db.raw_meta_data

    with open(date + "-" + hour + ".json") as f:
        for line in f.readlines():
            try:
                i += 1

                instance = json.loads(line.strip())


                if instance['type'] == "PushEvent":
                    instance_payload = instance['payload']
                    commit_message_list, commit_sha_list = filter_message(instance_payload)
                    if len(commit_message_list) != 0:
                        if my_collection.find_one({'repo_name': instance['repo']['name'].strip()}):
                            continue
                        if not branch_filtered(instance):

                            my_collection.insert_one({'repo_name': instance['repo']['name'].strip(),
                                                        'commit_id': commit_sha_list,
                                                        'commit_message': commit_message_list, 
                                                        'commit_date': instance['created_at']})
                            
                continue

            except Exception as e:
                logger.info(f"{i}th record has issue: repo name {instance['repo']['name']}")
           
                continue


def not_empty(record):
    """
    some attributes might be empty, discard the whole record if any one of them is absent.
    """
    for attribute in record:
        if attribute is None or attribute == '':
            return False

    return True


def extract_prev_sha_and_time(repo, file, current_commit_sha):
    """
    In the commit history on the main branch, grab the commit sha for the last version of the target file.
    """
    prev_sha, prev_time = "", ""
    current_commit = repo.get_commit(current_commit_sha)
    for commit in repo.get_commits(path=file, until=current_commit.commit.author.date):
        if commit.sha != current_commit_sha:
            prev_sha = commit.sha
            prev_time = commit.commit.author.date
            break

    return prev_sha, prev_time


def fetch_file(date, hour):
    """
    date in a format of year-month-day
    """
    subprocess.call(['./fetch_file.sh', date, str(hour)])


def delete_file(date, hour):
    subprocess.call(['./delete_file.sh', date, str(hour)])


def get_main_branch_sha(repo):
    sha = ""
    for branch in repo.get_branches():
        if branch.name == "main" or branch.name == "master":
            sha = branch.raw_data['commit']['sha']
            break

    return sha


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int)
    args = parser.parse_args()
    logger.add(f"output_{args.id}.log")
    g = initialize_configuration(args.id)

    start_date = datetime.date(2023, 2, 28)
    end_date = datetime.date(2015, 1, 1)
    days = (start_date - end_date).days
  
    for day in range(days+1):
        # if day % 6 != args.id:
        #     continue
        date = (start_date - datetime.timedelta(day)).strftime('%Y-%m-%d')
        for hour in reversed(range(24)):
            try:
                fetch_file(date, hour)
                logger.info(f"extracting data from {date} - {hour}")
                meta_data_extraction(date, str(hour), g, logger)
                delete_file(date, hour)
            except:
                logger.info(f"failed to extract data from {date} - {hour}")
                continue

    





