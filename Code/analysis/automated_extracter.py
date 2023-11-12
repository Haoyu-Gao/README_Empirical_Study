import markdown2
from collections import Counter, defaultdict
import pandas as pd
import subprocess
import os
from data_preparation import traverse_readmes
import json

from loguru import logger

def extract_header(file_name):
    """
    This function extract the headers of different sections in a markdown file.
    """
    with open(file_name, "r") as f:
        text = f.read()
    html = markdown2.markdown(text)
    
    # extract the headers
    headers = []
    for line in html.split("\n"):
        if line.startswith("<h"):
            headers.append(line)
    return headers


def extract_all_headers(file_name):
    """
    This function extract all headers in a markdown file.
    """
    headers = defaultdict(Counter)

    sampled_data = pd.read_excel(file_name)
    repo_names = sampled_data["repo_name"].tolist()
    
    for repo_name in repo_names:

        logger.info(repo_name)
        traverse_readmes(repo_name)
        # count the number of readme files
        count = 0
        for file in os.listdir("."):
            if file.startswith("readme") and file.endswith(".md"):
                count += 1      

        # extract the headers
        for i in range(1, count + 1):
            try:
                readme_file_name = f"readme_{i}.md"
                pacth_file_name = f"patch_parsed_{i}.json"

                headers_list = extract_header(readme_file_name)
                for header in headers_list:
                    headers[repo_name][header] += 1
                # remove the readme file
                subprocess.run(["rm", readme_file_name])
                subprocess.run(["rm", pacth_file_name])
            except:
                continue
        # write the headers to a json file
        with open("headers.json", "w") as f:
            json.dump(headers, f, indent=4)
        

def extract_patch_line(file_name):
    pass


if __name__ == "__main__":
    # print(extract_header("readme_1.md"))
    logger.add("extract_headers.log")
    extract_all_headers("selected_data.xlsx")
    # a = defaultdict(Counter)
