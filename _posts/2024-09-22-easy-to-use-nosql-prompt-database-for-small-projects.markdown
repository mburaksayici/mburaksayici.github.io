---
layout: post
title:  "Easy-to-use NoSQL Prompt Database for Small Projects"
date:   2024-09-22 00:00:10 +0300
categories: blog 
---

#### Ever needed a database for your LLM project that:

1. Your project isn't really big
2. You don't have to expose your prompts to business team, and even you are not really changing frequently.
3. Have 100s of prompts
4. You don't really do promp chaining/versioning/branching(wth are those terms lol).
5. You use agents but don't play with prompts to much
6. You just commit to git/bit to deploy, ie your CI/CD is simple as that, not complex
7. You mostly read from DB (you do actually)

And probably your strategy was keeping the prompts on .py files, so that you can change, play easily. Mine was [either.](https://github.com/mburaksayici/FinancialAdvisorGPT) But it comes with a cost that number of lines/weird complexity in python.

#### I need sth like SQLite for NoSQL 

I had used sqlite for one of the projects and has also seen that ChromaDB stores those vectors just in mssql. No external dependency, no need to deploy/make it up, perfect for small cases. Then I realise I wanted a database that:

1. SQLite for NoSQL, since we deal with prompts
2. No need to deploy and no need to add layer like having mongodb in docker-compose
3. Still I want to change prompts easily and commit to git/bit (Compare it with the case of MongoUI that you need to deploy the Mongo container, get into the MongoUI and change prompts)
4. Can organise prompts easily

#### Answer

After some search, here's the answer, TinyDB!

[![TinyDB](https://tinydb.readthedocs.io/en/latest/_static/logo.png)](https://tinydb.readthedocs.io)

Why Use TinyDB?

tiny: The current source code has 1800 lines of code (with about 40% documentation) and 1600 lines tests.
document oriented: Like MongoDB, you can store any document (represented as dict) in TinyDB.
optimized for your happiness: TinyDB is designed to be simple and fun to use by providing a simple and clean API.

written in pure Python: TinyDB neither needs an external server (as e.g. PyMongo) nor any dependencies from PyPI.

works on Python 3.5+ and PyPy: TinyDB works on all modern versions of Python and PyPy.
powerfully extensible: You can easily extend TinyDB by writing new storages or modify the behaviour of storages with Middlewares.


One bad thing is that manual changes to JSON should mistakenly break the format, but there are json-checker libraries, just add them to your CI/CD or test pipelines. 