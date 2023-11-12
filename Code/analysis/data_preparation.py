from pydriller import Repository, ModificationType
import csv
from github import Github
from credentials import *
import json
from markdown_it import MarkdownIt
from pprint import pprint
from collections import namedtuple
import numpy as np
import pandas as pd
from datetime import datetime
import difflib
from loguru import logger
import re
import nltk
import os

relevant_headers = set()
irrelevant_headers = set()
relevant_keywords = set()
with open ("annotated_headers/relevant_headers.txt", "r") as f:
    for line in f:
        relevant_headers.add(line.strip())
with open ("annotated_headers/irrelevant_headers.txt", "r") as f:
    for line in f:
        irrelevant_headers.add(line.strip())
with open ("annotated_headers/relevant_keywords.txt", "r") as f:
    for line in f:
        relevant_keywords.add(line.strip())



def get_readme_commits(repo_name):
    """
    This function return the readme commits for a given repository, as well as the readme file contents and all files modified.
    """
    repo_url = f"https://github.com/{repo_name}.git"
    repo_name = repo_name.replace('/', '_')

    readme_shas = []
    readme_commits = []
    changed_files = []
    changed_time = []
    authors = []

    for commit in Repository(repo_url, filepath='README.md').traverse_commits():
        readme_shas.append(commit.hash)
        readme_commits.append(commit.msg)
        changed_time.append(commit.committer_date)
        authors.append(commit.author.name)

    with open(f"{repo_name}.csv", "w") as f:
        writer = csv.writer(f)
        for i in range(len(readme_shas)):
            writer.writerow([readme_shas[i], readme_commits[i], changed_time[i], authors[i]])  


def get_issues(repo_name):
    """
    This function return the issue for a given repository.
    """
    g = Github(ACCESS_TOKEN_0)
    repo = g.get_repo(repo_name)
    issues = repo.get_issues(state='all')
    for issue in issues:
        print(issue.created_at)
        # print(issue.title)
        # print(issue.id)
        print(issue.number)


def get_readme(repo_name, sha, to_folder):
    """
    This function return the readme file contents for a given repository.
    """

    repo_url = f"https://github.com/{repo_name}.git"
    print(repo_url)
    for commit in Repository(repo_url, single=sha).traverse_commits():
        # get the readme file contents
        modified_files = commit.modified_files
        for modified_file in modified_files:
            if modified_file.filename == "README.md" and modified_file.new_path == "README.md":
                # print(modified_file.new_path)
                readme_curr = modified_file.source_code
                patch_parsed = modified_file.diff_parsed
                readme_prev = modified_file.source_code_before

        try:
            with open(f"{to_folder}/readme_curr.md", "w") as f:
                f.write(readme_curr)
            # patch_parsed is a dict, write it in json
            with open(f"{to_folder}/patch_parsed.json", "w") as f:
                json.dump(patch_parsed, f, indent=4)
            with open(f"{to_folder}/readme_prev.md", "w") as f:
                if readme_prev is not None:
                    f.write(readme_prev)
        except Exception as  e:
            print(e)
            continue    

def traverse_readmes(repo_name):
    count = 0
    repo_url = f"https://github.com/{repo_name}.git"
    for commit in Repository(repo_url, filepath="README.md").traverse_commits():
        try:
            count += 1
            modified_files = commit.modified_files
            for modified_file in modified_files:
                if modified_file.filename == "README.md":
    
                    readme = modified_file.source_code
                    patch = modified_file.diff
                    patch_parsed = modified_file.diff_parsed
            with open(f"readme_{count}.md", "w") as f:
                f.write(readme)

            
            with open(f"patch_parsed_{count}.json", "w") as f:
                json.dump(patch_parsed, f, indent=4)
        except Exception as  e:
            print(e)
            continue

def traverse_readmes_API(repo_name):
    count = 0
    g = Github(ACCESS_TOKEN_0)
    repo = g.get_repo(repo_name)
    for commit in repo.get_commits(path="README.md"):
        print(commit.sha)


def extract_sections(readme_file):
    sections = []
    # with open(readme_file, 'r') as file:
    #     readme_text = file.read()

    readme_text = readme_file   
    md = MarkdownIt()
    tokens = md.parse(readme_text)
    
    current_section = None
    for token in tokens:
        if token.type == 'heading_open':
            try:
                section_level = int(token.tag[1])
            except:
                section_level = 1
            section_title = tokens[tokens.index(token) + 1].content
            current_section = {'title': section_title, 'content': '', 'level': section_level, 'children': []}
            sections.append(current_section)
        elif current_section is not None:
            if token.type == 'inline':
                current_section['content'] += "\n\n" + token.content
            elif token.type == 'fence':
                content = token.content.strip()
                current_section['content'] += "\n\n" + "```" + token.info + "\n" + content + "\n```"
    
    for section in sections:
        section['content'] = section['content'].lstrip(section['title'])

    return sections


Section = namedtuple('Section', ['title', 'children'])

def construct_section_tree(sections, first_content=None):
    section_stack = []
    
    root = Section(title='', children=[])

    for section in sections:
        while len(section_stack) > 0 and section_stack[-1]['level'] >= section['level']:
            section_stack.pop()

        if len(section_stack) > 0:
            section_stack[-1]['children'].append(section)
        else:
            root.children.append(section)

        section_stack.append(section)
        

    return root.children

def traverse_section_tree(section):
    print('#' * section['level'], section['title'])
    for child in section['children']:
        traverse_section_tree(child)


def are_identical(section1, section2):
    """
    This function recursively checks if two sections are identical.
    """
    if section1['title'] != section2['title'] or \
        section1['level'] != section2['level'] or \
        section1['content'] != section2['content'] or \
        len(section1['children']) != len(section2['children']):
        return False
    for i in range(len(section1['children'])):
        if not are_identical(section1['children'][i], section2['children'][i]):
            return False
    return True

def get_changed_sections(top_section1, top_section2):
    # compare each children of top_section1 with top_section2, return the changed_sections
    changed_sections = []
    top_section1_children = [section['title'] for section in top_section1['children']]
    top_section2_children = [section['title'] for section in top_section2['children']]

    if len(top_section1_children) != len(top_section2_children):
        added_section =  list(set(top_section2_children) - set(top_section1_children))
        removed_section = list(set(top_section1_children) - set(top_section2_children))
        changed_section += added_section
        changed_section += removed_section
    
    else:
        for i in range(len(top_section1_children)):
            if not are_identical(top_section1['children'][i], top_section2['children'][i]):
                changed_sections.append(top_section1_children[i]['title'])

    return changed_sections


def contain_keywords(header, keywords):
    header = header.lower()
    words = nltk.word_tokenize(header)
    stemmer = nltk.stem.porter.PorterStemmer()
    flag = False
    for word in words:
        stemmed_word = stemmer.stem(word)
        if stemmed_word in keywords:
            flag = True
            break

    return flag

def compare_add_remove_section(headers1, headers2, rest_level):
    
    added_headers = []
    removed_headers = []
    all_irrelevant_updates = True

    # if len(headers1) != len(headers2):
    headers1_titles = [header['title'] for header in headers1]
    headers2_titles = [header['title'] for header in headers2]
    added_header = list(set(headers2_titles) - set(headers1_titles))
    removed_header = list(set(headers1_titles) - set(headers2_titles))
    for header in added_header:
        header = header.rstrip(":")
        if header.lower() in relevant_headers:
        # or contain_keywords(header, relevant_keywords):
            return True
        if header.lower() not in irrelevant_headers:
            added_headers.append(header)
            all_irrelevant_updates = False

    for header in removed_header:
        # header is a string, remove all non-alphanumeric characters
        header = header.rstrip(":")
        if header.lower() in relevant_headers or contain_keywords(header, relevant_keywords):
            return True
        if header.lower() not in irrelevant_headers:
            removed_headers.append(header)
            all_irrelevant_updates = False

    if not all_irrelevant_updates:
        for added_header in added_headers:
            try:
                # traverse through 2 levels of the children to see whether their added title are relevant
                header = headers2[headers2_titles.index(added_header)]
                if header['title'].lower() in relevant_headers or contain_keywords(header['title'], relevant_keywords):
                    return True
                for subchild in header['children']:
                    if subchild['title'].lower() in relevant_headers or contain_keywords(subchild['title'], relevant_keywords):
                            return True
            except:
                continue
        for removed_header in removed_headers:
            try:
                header = headers1[headers1_titles.index(removed_header)]
                if header['title'].lower() in relevant_headers or contain_keywords(subchild['title'], relevant_keywords):
                    return True
                for subchild in header['children']:
                    if subchild['title'].lower() in relevant_headers or contain_keywords(subchild['title'], relevant_keywords):
                            return True
            except:
                continue
        
    return None if all_irrelevant_updates == False else False
            

def compare_one_level(headers1, headers2, level=1):

    all_irrelevant_updates = True

    # deal with the case that the number of headers are different
    add_remove_flag = compare_add_remove_section(headers1, headers2, rest_level=4-level)
    if add_remove_flag == True:
        return True
  
    headers1_titles = [header['title'] for header in headers1]
    headers2_titles = [header['title'] for header in headers2]

    common_header = list(set(headers1_titles) & set(headers2_titles))

    # it might be the case that order has changed, we did not consider it as a change in this case??
    not_sure_headers = []
    not_sure_content = []

    for header in common_header:
        # index their header
        header1 = headers1[headers1_titles.index(header)]
        header2 = headers2[headers2_titles.index(header)]
        if not are_identical(header1, header2):
            header = header.rstrip(":")
            if header.lower() in relevant_headers or contain_keywords(header, relevant_keywords):
                return True
            if header.lower() not in irrelevant_headers:
                all_irrelevant_updates = False
                if len(header1['children']) != 0 and len(header2['children']) != 0:
                    not_sure_headers.append((header1['children'], header2['children']))
                    not_sure_content.append((header1['content'], header2['content']))
                
    if all_irrelevant_updates and add_remove_flag is not None:
        return False
    elif level == 4 or len(not_sure_headers) == 0:
        # no need to further compare
        return None
    else:
        all_irrelevant_updates = True
        # breakpoint()
        i = 0
        for header1, header2 in not_sure_headers:
            i += 1
            result = compare_one_level(header1, header2, level+1)
            if result == True:
                return True
            elif result is None:
                # content1, content2 = not_sure_content[i]
                # if content1 != content2:
                #     return None
                # else:
                all_irrelevant_updates = False
            else:
                continue
        
        if all_irrelevant_updates:
            return False
        else:
            return None
            
    # deal with the case that the number of headers are the same
    

def compare_readmes(file1, file2):
    """
    This function compares two readme files to determine whether the update 
    commit could be directly marked as relevant/ irrelevant.
    """
    all_irrelevant_update = True

    sections1 = extract_sections(file1)
    # breakpoint()
    sections2 = extract_sections(file2)
 

    
    # get the text before the first section
    text1 = ""
    text2 = ""
    if len(sections1) > 0:
        text1 = file1.split(int(sections1[0]['level']) * "#" + " " + sections1[0]['title'])[0]
    if len(sections2) > 0:
        text2 = file2.split(int(sections2[0]['level']) * "#" + " " + sections2[0]['title'])[0]
    if text1 != text2:
        all_irrelevant_update = False

    top_headers1 = construct_section_tree(sections1)
    top_headers2 = construct_section_tree(sections2)
    # breakpoint()
    return compare_one_level(top_headers1, top_headers2)


def iterate_readmes():
    data = pd.read_excel("selected_data.xlsx")
    repo_names = data['repo_name']
    for repo_name in repo_names:
        
        logger.info(f"Processing {repo_name}")
        shas = []
        messages = []
        results = []
        dates = []
        results = []
        url = f"https://github.com/{repo_name}.git"
        to_date = datetime(2023, 6, 1)
        for commit in Repository(url, filepath="README.md", to=to_date).traverse_commits():
            # get the current and prev readme files
            try:
                sha = commit.hash
                message = commit.msg
                date = commit.committer_date
                modified_files = commit.modified_files
                for modified_file in modified_files:
                    if modified_file.filename == "README.md" and modified_file.new_path == "README.md":
                        if modified_file.source_code is not None:
                            current_readme = modified_file.source_code
                        else:
                            current_readme = ""
                        if modified_file.content_before is not None:
                            prev_readme = modified_file.content_before.decode('utf-8')
                        else:
                            prev_readme = ""
                        result = compare_readmes(current_readme, prev_readme)
                
                if current_readme == "":
                    result = False
                shas.append(sha)
                messages.append(message)
                dates.append(date)
                results.append(result)
            except Exception as e:
                logger.info(e)
                continue
        
        repo_name = repo_name.replace('/', '@_@')
        with open(f"final_experiment4/{repo_name}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sha', 'message', 'date', 'result'])
            for i in range(len(shas)):
                writer.writerow([shas[i], messages[i], dates[i], results[i]])


def prepare_one_record(repo_name, sha, to_folder):
    get_readme(repo_name, sha, to_folder)
    repo_url = f"https://github.com/{repo_name}.git"
    added_files, deleted_files, modified_files = set(), set(), set()

    repo_name1 = repo_name.replace('/', '@_@')
    data = pd.read_csv(f"../commit_annotation/{repo_name1}.csv")
   
    # data['sha'] == sha gives the row that contains the sha, the last sha is the previous row
    try:
        last_sha = data.iloc[data[data['sha'] == sha].index[0] - 1]['sha'] 
    except:
        last_sha = None
    flag = False
    for commit in Repository(repo_url, to_commit=sha).traverse_commits():
        # ignore the last README modification commit.
        if flag:   
            for modified_file in commit.modified_files:
                if modified_file.change_type == ModificationType.ADD:
                    added_files.add(modified_file.new_path)
                elif modified_file.change_type == ModificationType.DELETE:
                    deleted_files.add(modified_file.old_path)
                elif modified_file.change_type == ModificationType.MODIFY:
                    modified_files.add(modified_file.new_path)
        
        if last_sha is None or commit.hash == last_sha:
            flag = True


    print(added_files, deleted_files, modified_files)




def prepare_samples():
    # first perform sampling on the relevant commits
    data = pd.read_csv("../commit_annotation/relevant_updates.csv")

    grouped = data.groupby('repo').count()

    min, lower, median, upper, max = grouped['sha'].describe()[['min', '25%', '50%', '75%', 'max']]
    bucket_1, bucket_2, bucket_3, bucket_4 = [], [], [], []
    
    for repo_name, df in grouped.iterrows():
        if df['sha'] <= lower:
            bucket_1.append(repo_name)
        elif lower < df['sha'] <= median:
            bucket_2.append(repo_name)
        elif median < df['sha'] <= upper:
            bucket_3.append(repo_name)
        else:
            bucket_4.append(repo_name)

    bucket1_all, bucket2_all, bucket3_all, bucket4_all = [], [], [], []
    for i in range(len(data)):
        if data['repo'].iloc[i] in bucket_1:
            bucket1_all.append([data['repo'].iloc[i], data['sha'].iloc[i]])
        elif data['repo'].iloc[i] in bucket_2:
            bucket2_all.append([data['repo'].iloc[i], data['sha'].iloc[i]])
        elif data['repo'].iloc[i] in bucket_3:
            bucket3_all.append([data['repo'].iloc[i], data['sha'].iloc[i]])
        else:
            bucket4_all.append([data['repo'].iloc[i], data['sha'].iloc[i]])

    print(len(bucket1_all))
    print(len(bucket2_all))
    print(len(bucket3_all))
    print(len(bucket4_all))

    # write to csv
    with open("../samples/bucket1.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['repo', 'sha'])
        for row in bucket1_all:
            writer.writerow(row)
    with open("../samples/bucket2.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['repo', 'sha'])
        for row in bucket2_all:
            writer.writerow(row)
    with open("../samples/bucket3.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['repo', 'sha'])
        for row in bucket3_all:
            writer.writerow(row)
    with open("../samples/bucket4.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['repo', 'sha'])
        for row in bucket4_all:
            writer.writerow(row)




if __name__ == "__main__":

    iterate_readmes()
