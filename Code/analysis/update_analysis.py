import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from github import Github
from pydriller import Repository, ModificationType
import markdown
import difflib
from credentials import *
import subprocess
import os
import csv
import datetime
from collections import defaultdict, Counter
import json
import re
import hashlib
from loguru import logger
import ast
import nltk 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import markdown2
from bs4 import BeautifulSoup
import string

stemmer = PorterStemmer()
lemmatiser = nltk.stem.WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
stop_words.add('http')

from data_preparation import extract_sections, construct_section_tree

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



def extract_relevant_updates():
    all_results = []
    for file in os.listdir("../commit_annotation"):
        if file.endswith(".csv"):
            data = pd.read_csv(f"../commit_annotation/{file}")
            file = file[:-4]
            file = re.sub(r'@_@', '/', file)
            results = data['result']
            shas = data['sha']
            for i in range(len(results)):
                if results[i] == True:
                    all_results.append((file, shas[i]))

    df = pd.DataFrame(all_results, columns=['repo', 'sha'])
    df.to_csv("../commit_annotation/relevant_updates.csv", index=False)


def repo_proportion():
    """
    This function calculates the proportion of repos that have at least one relevant update.
    """
    data = pd.read_csv("../commit_annotation/relevant_updates.csv")
    all_repos = pd.read_excel("selected_data.xlsx")
    repos = set(data['repo'])
    print(len(repos)/ len(all_repos))
    print(len(repos))
 


def repo_accumulate_proportion():
    data = pd.read_csv("../commit_annotation/relevant_updates.csv")
    # group by repo and sort descendingly by the number of relevant updates
    grouped = data.groupby('repo').count()
    grouped = grouped.sort_values(by=['sha'], ascending=False)

    stars = []
    count = []
    created_at = []


    # write to a csv file
    df = pd.DataFrame(np.column_stack([grouped.index, count, stars, created_at]), columns=['repo', 'count', 'stars', 'created_at'])
    df.to_csv("repo_sample_meta.csv", index=False)

    # plot a cumulative distribution
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(grouped)), grouped['sha'].cumsum())
    plt.xlabel("Number of repos")
    plt.ylabel("Number of relevant updates")
    plt.savefig("repo_accumulate_proportion.png")


def update_amount_statistics():
    """
    This function calculates the statistics for the number of *relevant* updates for each repo.
    The number of relevant updates varies a lot across repos.
    """
    g = Github(ACCESS_TOKEN_0)

    data = pd.read_csv("../commit_annotation/relevant_updates.csv")
    # for each repo, it has multiple relevant README updates "shas" for multiple rows, so we need to group by repo

    grouped = data.groupby('repo')
    # get the number of relevant updates for each repo

    counts = np.array(grouped.count())
    repo_names = list(grouped.groups.keys())
    # print(repo_names)
    print(counts.mean())
    print(counts.std())
    print(np.percentile(counts, 0))
    print(np.percentile(counts, 25))
    print(np.percentile(counts, 50))
    print(np.percentile(counts, 75))
    print(np.percentile(counts, 100))


    repo_contributors = []
    stars = []
    years = []
    for repo in repo_names:
        project = g.get_repo(repo)
        stars.append(project.stargazers_count)
        repo_contributors.append(project.get_contributors().totalCount)
        years.append(2023 - project.created_at.year + 1)


    stars = np.array(stars)
    repo_contributors = np.array(repo_contributors)
    years = np.array(years)
    # print(counts.shape)
    # print(stars.shape, repo_contributors.shape, years.shape)
    counts = counts.reshape(-1)
    stars = stars.reshape(-1)
    repo_contributors = repo_contributors.reshape(-1)
    years = years.reshape(-1)
    
    # compute the correlation between the number of relevant updates and the number of stars, size, contributors
    print(np.corrcoef((counts, stars, repo_contributors, years)))



def updates_size_contributors_stats():
    """
    This function quantitatively investigate the relationship between the number of relevant updates 
    and the size of the repo/ the number of contributors.
    """
    data = pd.read_csv("my_test_stats.csv")
    
    # normalise each column

    data['count'] = data['count'] / data['count'].max()
    data['size'] = data['size'] / data['size'].max()
    data['contributors'] = data['contributors'] / data['contributors'].max()

    counts = data['count']
    sizes = data['size']
    contributors = data['contributors']

    sns.set(style="ticks")
    g = sns.pairplot(data, diag_kind="kde")

    g.savefig("pairplot.png")   
    print(np.corrcoef((counts, sizes, contributors)))


def generate_hash(input_string):
    input_bytes = input_string.encode('utf-8')
    hashcode =  hashlib.sha256(input_bytes).hexdigest()
    # store the header-hashcode pair in a json file (first read then write and update)
    with open("header_hashcode.json", "r") as f:
        header_hashcode = json.load(f)
    header_hashcode.update({input_string: hashcode})
    with open("header_hashcode.json", "w") as f:
        json.dump(header_hashcode, f, indent=4)
    return hashcode


def construct_folder_structure(trees, folder_name):
    """
    given a list of trees, construct the folder structure from it.
    using depth travese to construct the folder structure
    """
    # create a folder, if exist, do nothing
    try:
        os.mkdir(folder_name)
    except:
        pass
    i = 1
    for tree in trees:
        title = tree['title']
        title = title.replace("/", "@")
        children = tree['children']
        content = tree['content']
        # create a folder for this section
        flag = False
        while not flag:
            try:
                os.mkdir(f"{folder_name}/{title}")
                flag = True
            except OSError as e:
                if e.errno == 36:
                    # file name too long
                    title_hash = generate_hash(title)
                    title = title_hash

                elif e.errno == 17:
                    # meaning there is multiple sections with the same name! This is bad! but sadly it exists.
                    title = title + f"{i}"
        # create a file for this section
        with open(f"{folder_name}/{title}/content.md", "w") as f:
            f.write(content)
        # recursively create folders for children
        construct_folder_structure(children, f"{folder_name}/{title}")
        i += 1

def extract_changed_headers():
    data = pd.read_csv("../commit_annotation/relevant_updates.csv")
    for i in range(len(data)):
        
        repo_name = data.iloc[i]['repo']
        
        sha = data.iloc[i]['sha']
        repo_url = f"https://github.com/{repo_name}.git"
        
        try:
            for commit in Repository(repo_url, single=sha).traverse_commits():
                commit_time = commit.committer_date
                for file in commit.modified_files:
                    if file.filename == "README.md" and file.new_path == "README.md":
                        prev = file.source_code_before
                        curr = file.source_code
                        prev_sections = extract_sections(prev) if prev is not None else []
                        curr_sections = extract_sections(curr) if curr is not None else []
                        prev_trees = construct_section_tree(prev_sections)
                        curr_trees = construct_section_tree(curr_sections)   
                        construct_folder_structure(prev_trees, 'v1')
                        construct_folder_structure(curr_trees, 'v2')
                        subprocess.call(["./construct_versions.sh"])
                        
                        added_sections = []
                        removed_sections = []
                        modified_sections = [] 
                        for idx, commit in enumerate(Repository("git_repo").traverse_commits()):
                            if idx == 1:
                                for file in commit.modified_files:
                                    if file.change_type == ModificationType.DELETE:
                                        removed_sections.append(file.old_path)
                                    elif file.change_type == ModificationType.ADD:
                                        added_sections.append(file.new_path)
                                    elif file.change_type == ModificationType.MODIFY:
                                        modified_sections.append(file.new_path)
       
                        with open("updated_headers_stats1.csv", "a") as f:
                            writer = csv.writer(f)
                            writer.writerow([repo_name, sha, added_sections, removed_sections, modified_sections, commit_time])
                        break
                # remove the processed folders.
                os.system("rm -rf v1")
                os.system("rm -rf v2")
                os.system("rm -rf git_repo")
        except:
            logger.info(f"Error in {repo_name} {sha}")

def update_temporal_clustering(threshold=5):
    """
    This function clusters the updates based on the time interval between two updates.
    Threshold indicating the hour interval.
    """
    data = pd.read_csv("updated_headers_stats.csv")
    data.columns = ['repo', 'sha', 'added_sections', 'removed_sections', 'modified_sections', 'time']
    grouped_data = data.groupby('repo')
    
    wait_time = []

    repo_clusters = defaultdict()
    for repo_data in grouped_data:
        # print(repo_data[1])
        # make the time clumn to be datetime type
        repo_name = repo_data[0]
        repo_data[1]['time'] = pd.to_datetime(repo_data[1]['time'])
        # sort the dataframe by time
        repo_data = repo_data[1].sort_values(by=['time'])
        
        # compute the time interval between two updates, convert it to hours
        for i in range(1, len(repo_data)):
            if (repo_data.iloc[i]['time'] - repo_data.iloc[i-1]['time']).total_seconds() == 0:
                print(repo_data.iloc[i]['time'], repo_data.iloc[i-1]['time'])
                print(repo_data.iloc[i]['sha'], repo_data.iloc[i-1]['sha'])
            wait_time.append((repo_data.iloc[i]['time'] - repo_data.iloc[i-1]['time']).total_seconds()/3600)

    print(min(wait_time))
    # box plot of the wait time, without outliers
    plt.boxplot(wait_time, showfliers=False)


    plt.savefig("wait_time.png")

    print(np.percentile(wait_time, 50))


def extract_temporal_clusters_stats():
    with open("temporal_clusters.json", "r") as f:
        clusters = json.load(f)

    number_of_clusters = []
    size_of_clusters = []
    multiple_updates_clusters = []
    for repo, clusters in clusters.items():
        number_of_clusters.append(len(clusters))
        for cluster in clusters:
            size_of_clusters.append(len(cluster))
        for cluster in clusters:
            if len(cluster) > 1:
                multiple_updates_clusters.append(cluster)


    # print out the statistics
    print(np.mean(number_of_clusters))
    print(np.std(number_of_clusters))
    print(np.percentile(number_of_clusters, 0))
    print(np.percentile(number_of_clusters, 25))
    print(np.percentile(number_of_clusters, 50))
    print(np.percentile(number_of_clusters, 75))
    print(np.percentile(number_of_clusters, 100))

    print(np.mean(size_of_clusters))
    print(np.std(size_of_clusters))
    print(np.percentile(size_of_clusters, 0))
    print(np.percentile(size_of_clusters, 25))
    print(np.percentile(size_of_clusters, 50))
    print(np.percentile(size_of_clusters, 75))
    print(np.percentile(size_of_clusters, 100))

    print(len(multiple_updates_clusters))
    print(np.mean([len(cluster) for cluster in multiple_updates_clusters]))
    print(np.std([len(cluster) for cluster in multiple_updates_clusters]))
    print(np.percentile([len(cluster) for cluster in multiple_updates_clusters], 0))
    print(np.percentile([len(cluster) for cluster in multiple_updates_clusters], 25))
    print(np.percentile([len(cluster) for cluster in multiple_updates_clusters], 50))
    print(np.percentile([len(cluster) for cluster in multiple_updates_clusters], 75))
    print(np.percentile([len(cluster) for cluster in multiple_updates_clusters], 100))



def test():
    for idx, commit in enumerate(Repository("git_repo").traverse_commits()):
        if idx == 1:
            print(commit.hash)
            for file in commit.modified_files:
                
                if file.change_type == ModificationType.DELETE:
                    print(file.old_path)
                else:
                    print(file.new_path)

def update_section_stats():
    """
    This function provides the overall statistics for how many sections are modified /added /deleted for each commit.
    """
    data = pd.read_csv("updated_headers_stats.csv")
    data.columns = ['repo', 'sha', 'added_sections', 'removed_sections', 'modified_sections', 'time']
    # convert added, removed, modified sections to list
    data['added_sections'] = data['added_sections'].apply(lambda x: ast.literal_eval(x))
    data['removed_sections'] = data['removed_sections'].apply(lambda x: ast.literal_eval(x))
    data['modified_sections'] = data['modified_sections'].apply(lambda x: ast.literal_eval(x))
    added_times, removed_times, modified_times = [], [], []
    for _, row in data.iterrows():
        
        added_times.append(len(row['added_sections']))
        removed_times.append(len(row['removed_sections']))
        modified_times.append(len(row['modified_sections']))
    
    print(f"added stats: avg: {np.mean(added_times)}, std: {np.std(added_times)}, min: {np.percentile(added_times, 0)}, 25 quantile: {np.percentile(added_times, 25)}, 50 quantile: {np.percentile(added_times, 50)}, 75 quantiles: {np.percentile(added_times, 75)}, max: {np.percentile(added_times, 100)}")
    print(f"removed stats: avg: {np.mean(removed_times)}, std: {np.std(removed_times)}, min: {np.percentile(removed_times, 0)}, 25 quantile: {np.percentile(removed_times, 25)}, 50 quantile: {np.percentile(removed_times, 50)}, 75 quantiles: {np.percentile(removed_times, 75)}, max: {np.percentile(removed_times, 100)}")
    print(f"modified stats: avg: {np.mean(modified_times)}, std: {np.std(modified_times)}, min: {np.percentile(modified_times, 0)}, 25 quantile: {np.percentile(modified_times, 25)}, 50 quantile: {np.percentile(modified_times, 50)}, 75 quantiles: {np.percentile(modified_times, 75)}, max: {np.percentile(modified_times, 100)}")

def update_section_stats_v2():
    """
    This function provides a finer granularity of headers(sections) that are modified /added /deleted for each commit.
    """
    data = pd.read_csv("updated_headers_stats1.csv")
    data.columns = ['repo', 'sha', 'added_sections', 'removed_sections', 'modified_sections', 'time']
    # convert added, removed, modified sections to list
    data['added_sections'] = data['added_sections'].apply(lambda x: ast.literal_eval(x))
    data['removed_sections'] = data['removed_sections'].apply(lambda x: ast.literal_eval(x))
    data['modified_sections'] = data['modified_sections'].apply(lambda x: ast.literal_eval(x))
    added, removed, modified = [], [], []
    for level in range(1, 5):
        added_times, removed_times, modified_times = [], [], []
        for _, row in data.iterrows():
            modified_headers = set()
            added_headers = set()
            removed_headers = set()

            for added_section in row['added_sections']:
                # the first one start with a dumb head "v1" and it ends with content.md should be removed.
                folder_hierarchies = added_section.split("/")[1:-1]  
                if len(folder_hierarchies) < level:
                    continue
                elif len(folder_hierarchies) == level:
                    added_headers.add(folder_hierarchies[level-1])
                else:
                    modified_headers.add(folder_hierarchies[level-1])
        
            for removed_section in row['removed_sections']:
                # the first one start with a dumb head "v1" and it ends with content.md should be removed.
                folder_hierarchies = removed_section.split("/")[1:-1]  
                if len(folder_hierarchies) < level:
                    if level == 1:
                        continue
                elif len(folder_hierarchies) == level:
                    removed_headers.add(folder_hierarchies[level-1])
                else:
                    modified_headers.add(folder_hierarchies[level-1])
                    
            for modified_section in row['modified_sections']:
                # the first one start with a dumb head "v1" and it ends with content.md should be removed.
                folder_hierarchies = modified_section.split("/")[1:-1]  
                if len(folder_hierarchies) < level:
                    continue
                else:
                    modified_headers.add(folder_hierarchies[level-1])
            added_times.append(len(added_headers))
            removed_times.append(len(removed_headers))
            modified_times.append(len(modified_headers))
        added.append(added_times)
        removed.append(removed_times)
        modified.append(modified_times)



    # there are 4 levels of headers, each level has three stats: added, removed, modified
    # in the plot (only with one figure), each level of added, removed and modified are grouped together

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(4):
        axs[i].boxplot([added[i], removed[i], modified[i]], labels=['added', 'removed', 'modified'])
        axs[i].set_title(f"level {i+1}")
        print(f"added stats: avg: {np.mean(added[i])}, std: {np.std(added[i])}, min: {np.percentile(added[i], 0)}, 25 quantile: {np.percentile(added[i], 25)}, 50 quantile: {np.percentile(added[i], 50)}, 75 quantiles: {np.percentile(added[i], 75)}, max: {np.percentile(added[i], 100)}")
        print(f"removed stats: avg: {np.mean(removed[i])}, std: {np.std(removed[i])}, min: {np.percentile(removed[i], 0)}, 25 quantile: {np.percentile(removed[i], 25)}, 50 quantile: {np.percentile(removed[i], 50)}, 75 quantiles: {np.percentile(removed[i], 75)}, max: {np.percentile(removed[i], 100)}")
        print(f"modified stats: avg: {np.mean(modified[i])}, std: {np.std(modified[i])}, min: {np.percentile(modified[i], 0)}, 25 quantile: {np.percentile(modified[i], 25)}, 50 quantile: {np.percentile(modified[i], 50)}, 75 quantiles: {np.percentile(modified[i], 75)}, max: {np.percentile(modified[i], 100)}")


    plt.savefig("updated_headers_stats_v2.png")

      
        


def update_section_stats_v3(level=1):
    """
    This function calculates the most frequently modified/ added/ deleted headers
    """ 
    data = pd.read_csv("updated_headers_stats.csv")
    data.columns = ['repo', 'sha', 'added_sections', 'removed_sections', 'modified_sections', 'time']
    # convert added, removed, modified sections to list
    data['added_sections'] = data['added_sections'].apply(lambda x: ast.literal_eval(x))
    data['removed_sections'] = data['removed_sections'].apply(lambda x: ast.literal_eval(x))
    data['modified_sections'] = data['modified_sections'].apply(lambda x: ast.literal_eval(x))
    
    added_counter, removed_counter, modified_counter = defaultdict(), defaultdict(), defaultdict()

    # count the repo occurence for the data
    repo_counter = Counter()
    for _, row in data.iterrows():
        repo_counter += Counter([row['repo']])

    for _, row in data.iterrows():
        modified_headers = set()
        added_headers = set()
        removed_headers = set()
        for added_section in row['added_sections']:
            # the first one start with a dumb head "v1" and it ends with content.md should be removed.
            folder_hierarchies = added_section.split("/")[1:-1]  
            if len(folder_hierarchies) < level:
                continue
            elif len(folder_hierarchies) == level:
                header = folder_hierarchies[-1].lower()
                lemmatised_words = [lemmatiser.lemmatize(word) for word in word_tokenize(header)]
                # remove stop words and non-alphabetic words
                lemmatised_words = [word for word in lemmatised_words if word not in stop_words and word.isalpha()]
                added_headers = added_headers.union(lemmatised_words)
            else:
                header = folder_hierarchies[level-1].lower()
                lemmatised_words = [lemmatiser.lemmatize(word) for word in word_tokenize(header)]
                lemmatised_words = [word for word in lemmatised_words if word not in stop_words and word.isalpha()]
                modified_headers = modified_headers.union(lemmatised_words)
    
        for removed_section in row['removed_sections']:
            # the first one start with a dumb head "v1" and it ends with content.md should be removed.
            folder_hierarchies = removed_section.split("/")[1:-1]  
            if len(folder_hierarchies) < level:
                continue
            elif len(folder_hierarchies) == level:
                header = folder_hierarchies[-1].lower()
                lemmatised_words = [lemmatiser.lemmatize(word) for word in word_tokenize(header)]
                lemmatised_words = [word for word in lemmatised_words if word not in stop_words and word.isalpha()]
                removed_headers = removed_headers.union(lemmatised_words)
            else:
                header = folder_hierarchies[level-1].lower()
                lemmatised_words = [lemmatiser.lemmatize(word) for word in word_tokenize(header)]
                lemmatised_words = [word for word in lemmatised_words if word not in stop_words and word.isalpha()]
                modified_headers = modified_headers.union(lemmatised_words)
                
        for modified_section in row['modified_sections']:
            # the first one start with a dumb head "v1" and it ends with content.md should be removed.
            folder_hierarchies = modified_section.split("/")[1:-1]  
            if len(folder_hierarchies) < level:
                continue
            else:
                header = folder_hierarchies[level-1].lower()
                lemmatised_words = [lemmatiser.lemmatize(word) for word in word_tokenize(header)]
                lemmatised_words = [word for word in lemmatised_words if word not in stop_words and word.isalpha()]
                modified_headers = modified_headers.union(lemmatised_words)
        
        for header in added_headers:
            added_counter[header] = added_counter.get(header, 0) + 1 / repo_counter[row['repo']]
        for header in removed_headers:
            removed_counter[header] = removed_counter.get(header, 0) + 1 / repo_counter[row['repo']]
        for header in modified_headers:
            modified_counter[header] = modified_counter.get(header, 0) + 1 / repo_counter[row['repo']]
        # added_counter.
        # removed_counter += Counter(removed_headers)
        # modified_counter += Counter(modified_headers)

    # print the top 20 most frequently modified/ added/ deleted headers, without their frequencies displayed

    # added_sorted = sorted(added_counter.items(), key=lambda x: x[1], reverse=True)[:50]
    # removed_sorted = sorted(removed_counter.items(), key=lambda x: x[1], reverse=True)[:50]
    # modified_sorted = sorted(modified_counter.items(), key=lambda x: x[1], reverse=True)[:50]


    return added_counter, removed_counter, modified_counter
  

def section_text_changes(level=2, section_header=["Installation"], mode="modified"):
    """
    This function goes through the content.md of the section at the given level, as well as accumulate the 
    sub-sections content.md underneath it.
    Textual changes includes: 
    1. number of words
    2. number of sentences
    3. number of paragraphs
    4. number of code blocks
    5. number of links
    6. number of images
    7. lines of code in code blocks
    """
    data = pd.read_csv("updated_headers_stats1.csv")
    data.columns = ['repo', 'sha', 'added_sections', 'removed_sections', 'modified_sections', 'time']
    # convert added, removed, modified sections to list
    data['added_sections'] = data['added_sections'].apply(lambda x: ast.literal_eval(x))
    data['removed_sections'] = data['removed_sections'].apply(lambda x: ast.literal_eval(x))
    data['modified_sections'] = data['modified_sections'].apply(lambda x: ast.literal_eval(x))

    column = mode + "_sections"
    prev_stats, curr_stats = defaultdict(), defaultdict()

    for i in range(len(data)):
        sections = data.iloc[i][column]
        print(i)
        for section in sections:
            try:
                if section.split('/')[1:-1][level-1] == section_header:
                    repo_url = f"https://github.com/{data.iloc[i]['repo']}.git"
                    sha = data.iloc[i]['sha']
                    for commit in Repository(repo_url, single=sha).traverse_commits():
                        for file in commit.modified_files:
                            if file.filename == "README.md":

                                with open("temp.md", "w") as f:
                                    f.write(file.source_code)
                                prev = file.source_code_before
                                curr = file.source_code
                                prev_sections = extract_sections(prev) if prev is not None else []
                                curr_sections = extract_sections(curr) if curr is not None else []

                                prev_trees = construct_section_tree(prev_sections)
                                curr_trees = construct_section_tree(curr_sections)   
                                construct_folder_structure(prev_trees, 'v1')
                                construct_folder_structure(curr_trees, 'v2')
                                
                                if mode == "added":
                                    pass
                                elif mode == "removed":
                                    pass
                                else:
                                    # first find the section at the given level
                                    v1_folder = f"v1/{'/'.join(section.split('/')[1:-1][:level])}"
                                    v2_folder = f"v2/{'/'.join(section.split('/')[1:-1][:level])}"
                                    
                                    # then accumulate the sub-sections content.md underneath it
                                    v1_text = accumulate_text(v1_folder)
                                    v2_text = accumulate_text(v2_folder)
                                    # then calculate the statistics
                                    v1_stats = calculate_textual_statistics(v1_text)
                                    v2_stats = calculate_textual_statistics(v2_text)
                                    update_stats(prev_stats, v1_stats)
                                    update_stats(curr_stats, v2_stats)

                                    os.rmdir("v1")
                                    os.rmdir("v2")
                            
            except:
                continue
    
    for key, value in prev_stats.items():
        print(f"prev {key}: {np.average(value)}")

    for key, value in curr_stats.items():
        print(f"curr {key}: {np.average(value)}")

        
def accumulate_text(folder):
    """
    Concatenate all the content.md underneath the given folder, recursively under all sub folders.
    """
    text = ""
    try:
        for file in os.listdir(folder):
            if file == "content.md":
                with open(os.path.join(folder, file), 'r') as f:
                    text += "\n\n" + f.read()
            else:
                text += accumulate_text(os.path.join(folder, file))
    except Exception as e:
        print(e)
    
    return text
        

def calculate_textual_statistics(text):
    """
    mask the code blocks and links, then calculate the statistics
    """
    # mask the code blocks
    
    # mask the links, code blocks to <code_small>, <code_large> and <link>
    html_text = markdown2.markdown(text, extras=['fenced-code-blocks', 'tables'])
 
    soup = BeautifulSoup(html_text, 'html.parser')

    pre_tags = soup.findAll('pre')
    for tag in pre_tags:
        tag.string = "code_large."

    code_tags = soup.findAll('code')
    for tag in code_tags:
        tag.string = "code_small"
    
    table_tags = soup.findAll('table')
    for tag in table_tags:
        tag.string = "table"

    url_tags = soup.findAll('a')
    for tag in url_tags:
        url_text = tag.text.strip()
        tag.replace_with(f'{url_text} url')

    # word tokens should be calculated without the punctuations
    
    word_tokens = len([word for word in word_tokenize(soup.get_text()) if word not in string.punctuation])
    sent_tokens = len(sent_tokenize(soup.get_text()))
  
    para_tokens = len(soup.get_text().split('\n\n'))
    code_blocks = len(soup.findAll('code'))
    links = len(soup.findAll('a'))
    images = len(soup.findAll('img'))
    code_lines = len(soup.findAll('pre'))
    tables = len(soup.findAll('table'))

    return {
        "word_tokens": word_tokens,
        "sent_tokens": sent_tokens,
        "para_tokens": para_tokens,
        "code_blocks": code_blocks,
        "links": links,
        "images": images,
        "code_lines": code_lines,
        "tables": tables
    }


def update_stats(overall_stats, tmp_stats):
    for key in tmp_stats:
        if key not in overall_stats:
            overall_stats[key] = []
        overall_stats[key].append(tmp_stats[key])


def sample_stategy_test():
    # this is only a test function, I will delete it later
    data = pd.read_csv("repo_sample_meta.csv")

    # scatter plot for stars vs number of commits, log scale for the x axis
    plt.xscale('log')
    plt.scatter(data['stars'], data['count'])
    plt.xlabel("stars")
    plt.ylabel("number of commits")
    plt.savefig("stars_vs_commits.png")



if __name__ == "__main__":
  

    for i in range(1, 3):
        header_weights = defaultdict()
        added_counter, removed_counter, modified_counter = update_section_stats_v3(level=i)

        added_sections = sorted(added_counter.items(), key=lambda x: x[1], reverse=True)[:50]
        removed_sections = sorted(removed_counter.items(), key=lambda x: x[1], reverse=True)[:50]
        modified_sections = sorted(modified_counter.items(), key=lambda x: x[1], reverse=True)[:50]

        data = []
        for header, weight in added_sections:
            header_weights[header] = header_weights.get(header, 0) + weight
        for header, weight in removed_sections:
            header_weights[header] = header_weights.get(header, 0) + weight
        for header, weight in modified_sections:
            header_weights[header] = header_weights.get(header, 0) + weight
        header_weights = sorted(header_weights.items(), key=lambda x: x[1], reverse=True)
        # only get the first 20 headers
        header_weights = header_weights[:20]
        # print(header_weights)
        headers = [x[0] for x in header_weights]
        add_weight = [added_counter.get(x, 0) for x in headers]
        remove_weight = [removed_counter.get(x, 0) for x in headers]
        modified_weight = [modified_counter.get(x, 0) for x in headers]
        data.append(add_weight)
        data.append(remove_weight)
        data.append(modified_weight)
        data = np.array(data)

        data = np.log(data+1)

        # plt.figure(figsize=(16, 12))
        sns.set(rc = {'figure.figsize':(16, 12)})
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        sns.set()
        cmap = sns.cm.rocket_r
        ax = sns.heatmap(data, cmap=cmap, xticklabels=headers, yticklabels=["added", "removed", "modified"],
                         vmin=0, vmax=4, cbar_kws={"orientation": "horizontal", "location": "top"})
        
        plt.gca().set_aspect(3)

        # Set x-axis and y-axis labels and adjust their properties
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontweight="bold")

        plt.title(f"Level {i}", fontsize=16, fontweight="bold")
        plt.savefig(f"word_changes_level_{i}.pdf", bbox_inches='tight')
        plt.close()
        

