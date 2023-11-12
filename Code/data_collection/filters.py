import json
import nltk

from credential import *
from configurations import TARGET_FILES, TARGET_LANGUAGES, KEYWORDS

headers = {f'Authorization': f"Token {ACCESS_TOKEN_1}"}

lemmatiser = nltk.WordNetLemmatizer()
tokenizer = nltk.SpaceTokenizer()


def star_language_filtered(repo):
    """
    Filter out toy repositories with less than 20 stars and filter out repositories not in the target language set
    """
    if repo.language is None:
        return True
    elif repo.stargazers_count >= 20 and repo.language.strip().lower() in TARGET_LANGUAGES:
        return False

    return True


def branch_filtered(instance):
    """
    Filter out commits that are not performed on the main branch.
    """
    branch = instance['payload']['ref'].split('/')[-1]
    if branch == "main" or branch == "master":
        return False
    else:
        return True


def filter_message(payload):
    """
    Go through all commits in one push, get those commits with messages that are interesting
    """
    returned_commits_message, returned_commits_shas = [], []
    commits = payload['commits']
    for commit in commits:
        message = commit['message'].lower().strip()

        if message.endswith('.'):
            message = message.rstrip('.')
        message_without_punct = message.replace(',', '').replace('!', '').replace('?', '')

        message_tokens = tokenizer.tokenize(message_without_punct)

        for token in message_tokens:
            token = token.lower().strip().rstrip('.')
            lemma = lemmatiser.lemmatize(token, "v")
            # lemmatise the word and compare with our keyword set.
            if lemma == token:
                lemma = lemmatiser.lemmatize(token, "n")
                if lemma == token:
                    lemma = lemmatiser.lemmatize(token, "a")
            if lemma in KEYWORDS:
                returned_commits_message.append(commit['message'])
                returned_commits_shas.append(commit['sha'])
                break

    return returned_commits_message, returned_commits_shas


def filter_file(repo, commit_message_list, commit_sha_list):
    """
    Filter based on the commit files categories, only stick to README.md.
    Also, we only collect the files located in the normal position, i.e., under root dir.

    When commits contain many file changes, the commit message tend to excessively long, containing noise (e.g.,
    a code file changing with commit message falling in the keyword set) that make the analysis difficult.
    In order to mitigate this, we only focus on commits that only change one target file and no other files.
    """
    msgs, raw_urls, patches, files, shas = [], [], [], [], []
    for i in range(len(commit_sha_list)):
        commit_message, commit_sha = commit_message_list[i], commit_sha_list[i]
        # print(repo)
        #
        # print(commit_sha)
        # print(commit_message)
        commit = repo.get_commit(sha=commit_sha)

        commit_content = commit.raw_data
        modified_files = commit_content['files']

        if len(modified_files) > 1:
            continue

        for file in modified_files:
            if file['filename'].lower().strip() in TARGET_FILES:
                msgs.append(commit_message)
                raw_urls.append(file['raw_url'])
                patches.append(file['patch'])
                files.append(file['filename'])
        shas.append(commit_content['sha'])

    return msgs, raw_urls, patches, files, shas