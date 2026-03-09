#!/bin/bash

git reset --hard HEAD~1

git checkout main
git fetch upstream main
git pull origin main
git checkout srebot
git rebase main
# git merge --abort
git merge feat/add-slash-models-and-status-endpoints