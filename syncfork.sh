#!/bin/bash
git reset --hard HEAD~2
git checkout main
git fetch upstream main
git pull origin main
git checkout srebot
git rebase main
git merge feature/own-skill-and-config
# git merge --abort
# git merge feat/add-slash-models-and-status-endpoints