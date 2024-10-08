---
layout: post
title:  "Building an Agentic RAG Application, a startup POC"
date:   2024-08-29 14:49:10 +0300
categories: blog 
tags: llms llm 
---

I was approached by a successful CEO from Turkey, asking for LLM application on some confidential financial domain. And it's a startup world, lots to talk on this later but we didn't continue, I wish him a good luck! 

I would like to share my experiences building its backend and LLMs. It's a fairly new area, and writing AI Agents from scratch (I'm privileged to code/maintain MLOps applications from scratch due to the last two company) is really helpful. 

To be honest, Langchain-like libraries are not good at all, don't expect pytorch level code/interface quality over there. Just click this google search [link](https://www.google.com/search?q=reddit+langchain+sucks), and you'll learn. Hope they'll get better! 

So you can ignore the domain, ignore why I haven't used Langchain in depth.

* Table of contents
{:toc}



### FinancialAdvisorGPT

{% raw %}

<video controls width=500 height= 400>
  <source src="https://github.com/mburaksayici/FinancialAdvisorGPT/assets/25187211/8618c9fb-fbf7-49ff-867d-2a15de47f4f8"  type="video/mp4">
</video>

{% endraw %}


FinancialAdvisorGPT is a boilerplate project designed for RAG (Retriever-Augmented Generation) and LLM (Large Language Model) applications in financial analysis. Built on a technology stack including MongoDB, MongoDB VectorDB, Chroma, FastAPI, Langchain, and React submodule for UI,  it offers a framework for developers to implement and customize RAG+LLM projects. Leveraging parallelized data pipelines, FinancialAdvisorGPT processes and integrates various data sources such as stock data, news, SEC filings, and local PDFs. With Mistral-Tiny and Mistral-Small LLM models handling natural language tasks, FinancialAdvisorGPT facilitates the generation of high-quality financial reports. Development challenges, including Langchain's complexity, are acknowledged, with ongoing efforts focused on enhancing functionality and performance for RAG+LLM applications in financial analysis.


Based on MongoDB+MongoDB VectorDB(TBD)+Chroma+FastAPI+Langchain+Mistral+Redis(TBD), plus react in a forked submodule project. RAG project that reads stock data, news, SEC filings (or your local pdfs) and creates high quality reports that is able to  reference and links to the sources without hallucination.

### Installment

You can learn about it on [here](https://github.com/mburaksayici/FinancialAdvisorGPT).

### Database Choice

Although i don't utilise it atm, database choice is mongodb latest version with its Vector DB feature.
Although I use Chromadb for now, I don't really believe the options like chromadb (which is sqlite) is necessary, big DB companies will offer faster Vector DB features, and they'll be better as well.

I also have done Vector similarity search in Gesund Inc., its just keeping arrays and retrieving it. Nothing fancy but feels you very intelligent, just do it!

### Cache Choice

Although I haven't activated it as of the date of 22 March, 2024, Redis to cache conversations between user and LLM architecture is extremely needed.

### Parallelization of Data Pipelines

RAG takes time.  Data acquisition, augmentation wrt the new data, feeding LLMs with new data takes time. Thus, each independent pipeline is parallelised. 

The RAG pipeline diagram below, DataPipelines are responsible for both generating questions to search for. DBRetrieval creates 10 questions given user prompt, then do vector search on DB which extracted from document. StockDataRetrieval creates 10 API request payloads given user prompt oversimplified like {"tinker":"TSL", "data": "MARKETCAP"}, then do API calls to get live data. Search of 10 questions, all parallelised.  It takes 45 secs to analyse 20 pdfs (30 pages on average), stock data of 2 companies and 10 news with parallelisation. Otherwise, it takes 3-4 mins to create example report above.

### RAG Pipeline

RAG pipeline, consist of 3 elements at the moment, that gathers data from pdfs, Stock Data and News data. PDF data is extracted to ChromaDB(but I plan to move it to MongoDB+Mongo Vector), stock and news data is gathered from online sources (API keys needed). 


![Baslksz_Diyagram drawio-2](https://github.com/mburaksayici/finsean/assets/25187211/07ab079d-3da0-41f6-8796-0fcf884b6d7e)

- One good thing is that, inherently, LLM decides if you need stock data. If user prompts "What happened to Tesla's EV fabric investment on Romania, is it deployed?", Data pipelines decide if this question needs to be answered by News or Stock Data. Inherently, if you look at the StockDataChain class, you'll observe that the system prompt includes "Give me empty list if question is not related to stock data.". Or if the user question is related to P/B value of Ford, News data won't search for it, again with similar manner.

- Although at the moment I only create a long report at the first message, then you only talk with the conversation history in the current codebase,  during the conversation not all the questions require searching all data sources. Here's the diagram i drawn. This has technical challenges that i plan to solve but not solved yet. Simply one LLM should decide if you need to access to all sources for new user prompt.


<img width="899" alt="Ekran Resmi 2024-03-23 02 27 16" src="https://github.com/mburaksayici/finsean/assets/25187211/b6257f72-69b1-436a-af18-d20389195965">

Red boxes are user prompt, green boxes are the answer of Responder LLM. Check the arrows in the upper green box. User Question(prompt) requires retrieval to all data sources. However, at the middle green box, user question(prompt) doesn't require any sources, question may be solved with the already-analysed data, so pipeline is not executed. 


### Problems on RAG Pipeline
1. LLM is not able to focus on large documents, in fact it focuses on the first paragraph[ Lost in The Middle Paper : https://arxiv.org/abs/2307.03172].
   
   Where you place the relevant information is (at least at 2024 March) very important. Solution : It can be solved with having multiple Responder LLMs(see pipeline) that each shares some part of the retrieved data and each responsible for creating one "section" of the report. 

2. One of the biggest problem of the current RAG pipeline is that, context lenght(as of the date of March 2024). Long textual data sources wont fit to single LLM. I have overcome this by 2 solution:

   a. Summarizer LLM: It's not showing up in the pipeline images, however, assume you would like to know why Snowflake stocks are down lately. So News Pipeline will try to gather latest news on Snowflake. It'll consume (let's say) 100 news, each having introductory paragraphs, conclusion, some editorial info. You wont need that in your augmented results. So, Summarizer LLM inherently summarizes all news to the 3-4 sentences, then augmented summary is embedded to your final report to be sent to Responder LLM.

   b. Although not implemented yet, multiple Responder LLMs(see pipeline) that each shares some part of the retrieved data and each responsible for creating one "section" of the report.
   
### LLM Model

Mistral-Tiny and Mistral-Small is used. Mistral-Tiny not always following the instructions, small is better.
Big/better models are also error-prone. Chip Huyen's blog has great insights and production experiences that I'll summarize, but the most interesting one is that :
"A couple of people who’ve worked with LLMs for years told me that they just accepted this ambiguity and built their workflows around that. It’s a different mindset compared to developing deterministic programs, but not something impossible to get used to."
So it means, sometimes you can do multiple requests to get what you need from LLM.  One case is that in StockDataChain request, I request python dict directly from LLM, and I do eval(str) which LLM sometimes doesn't produce in python-syntax. Retry again, it'll work. 

Multiple models can be used, for small tasks, cheap LLMs can be used. Or, some local small LLM models are also OK to use. Or, first  trial with small model and second trial with expensive models are also fine.

![Ekran_Resmi_2024-02-23_12 09 22](https://github.com/mburaksayici/finsean/assets/25187211/ac98014f-cfdb-4692-bdeb-fd642341d328)


### Experience on Langchain
As of the date of 23 March 2024, it sucks. After suffering with Langchain, i randomly typed "langchain sucks" to the Google because of boredom and was happy to coincidentially  confirm that it sucks from Redditors.

My comment is simply : Langchain python library is overengineered without documentation, which makes it complex. I won't give other details, here are the sources: 
https://analyticsindiamag.com/langchain-is-garbage-software/
https://www.reddit.com/r/datascienceproject/comments/16o246h/is_it_just_me_or_is_langchain_is_too_complicated/

Just search for "langchain is complex" in google to see what i mean.

### Streaming, and a lesson from Devin

Streaming is painful, at least for me atm. Serving the streaming API, encapsulating it on FASTAPI and sending the response as a streaming process, I just didn't focus too much on that. But one thing is interesting to discuss.

As I've said, inference takes time, and it takes much more time when you have RAG pipelines to acquire data. It may take very long if you would like to have more detailed responses.

Look what Devin does: https://www.youtube.com/watch?v=fjHtjT7GO1c

Think of this : You ask Devin to give you a Python project with mongodb+flask app that you order pizza. It'll initialise the project, do google search, execute and fix errors. It'll take may be hours to be done with it, right? How do you fool the user then? Devin shows all RAG pipelines in action. It executes selenium search in Google, it runs the code, it shows that errors are there, etc. etc. So once you inform user that "Hey I'm AI, and I'm working on it" and if you make user feel like a pissed-off boss who tracks the new employees computer to annoy, they'll enjoy. Sad but true. This gives UI-UX insights.

### What needs to be done

- Tracking the LLM pipelines. I wouldn't trust langchain for that atm.
- Tracking the data pipelines for query. Each answer needs to be tracked. Not all questions require to get all data from all sources. Some questions may be relevant to the conversation history as well. 

- [x] Chat Memory
- [x] News search through augmentation
- [x] Stock data acquisition through augmentation
- [x] Chunking local pdfs to Vector DB
- [x] Parallelization of RAG pipelines 
- [x] Vector DB (Chroma)
- [x] Chats stored in DB
- [x] Source citing
- [ ] Conversation Map through networkx (See chat diagram with black background below)
- [ ] Cost management
- [ ] Mongo Vector DB 
- [ ] Redis Cacheing for RAG
- [ ] Redis Cacheing for conversations
- [ ] Mongodb 


#### Some more insights

In addition to all of that, based on my experience on this project and [Chip Huyen's perfect blog.](https://huyenchip.com/2023/04/11/llm-engineering.html)

1. Ambiguous output format : there’s no guarantee that the outputs will always follow this format .
2. You can force an LLM to give the same response by setting temperature = 0, which is, in general, a good practice. While it mostly solves the consistency problem, it doesn’t inspire trust in the system.
3.  A couple of people who’ve worked with LLMs for years told me that they just accepted this ambiguity and built their workflows around that. It’s a different mindset compared to developing deterministic programs, but not something impossible to get used to. !!!
4. Prompt versioning : i plan to do that as well.
5. Prompt optimization : A research area.
6. Input tokens can be processed in parallel, which means that input length shouldn’t affect the latency that much. This principle is applied.
7. "The impossibility of cost + latency analysis for LLMs : The LLM application world is moving so fast that any cost + latency analysis is bound to go outdated quickly. Matt Ross, a senior manager of applied research at Scribd, told me that the estimated API cost for his use cases has gone down two orders of magnitude over the last year. Latency has significantly decreased as well."
8. "Prompt tuning- finetuning. : increase the number of examples, finetuning will give better model performance than prompting." It's same for other ml models but nice to read this exp.
9. Finetuning with distillation : a research area.
10. "If 2021 was the year of graph databases, 2023 is the year of vector databases." we 'll see.. Old-school DBs are also offering vector db feature since its just a list of numbers(vectors in DNNs).
11. One argument I often hear is that prompt rewriting shouldn’t be a problem because:
"Newer models should only work better than existing models." I’m not convinced about this. Newer models might, overall, be better, but there will be use cases for which newer models are worse.
12. "Experiments with prompts are fast and cheap, as we discussed in the section Cost." : While I agree with this argument, a big challenge I see in MLOps today is that there’s a lack of centralized knowledge for model logic, feature logic, prompts, etc. 
14. Control flows: sequential, parallel, if, for loop:  Shared the image above.  
15. "For example, if you want your agent to choose between three actions search, SQL executor, and Chat, you might explain how it should choose one of these actions as follows (very approximate), In other words, you can use LLMs to decide the condition of the control flow! " . Currently used in this project. Like :  "IF DO WE NEED STOCK DATA FOR A QUESTION, LLM DECIDES THAT IN THE CODEBASE!"

