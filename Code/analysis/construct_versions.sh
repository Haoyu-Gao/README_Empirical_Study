#!/bin/bash

# create a version control repository
mkdir git_repo

cp -r v1/ git_repo/
cd git_repo
git init
git add *
git commit -m "v1"

cp -r ../v2/ .
rm -rf v1/
cp -r v2/ v1/
rm -rf v2/
git add *
git commit -m "v2"
# the above process gives two vesions of the file, which we could use to extract changes



