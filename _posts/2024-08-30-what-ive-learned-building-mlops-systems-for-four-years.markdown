---
layout: post
title:  "What I've learned building MLOps systems for four years"
date:   2024-08-30 00:00:10 +0300
categories: blog 
---

As title says, I've  worked on building MLOps systems for nearly four years. Things run amazingly fast, and as a person that is also four years experienced guy, I kinda feel like I've tried to not drown by the new techs in deep learning(LLMs), trying to adapt software engineering, try to catch good positions in good companies remotely(that sucks), and more. 


Check my [LinkedIn profile](https://www.linkedin.com/in/mehmet-burak-sayici-a45294126/), this blog post will make much more sense, and it's fun to stalk as always. I had some experience on two stealth mode startups on LLMs, [check this post.]({% link _posts/2024-08-29-building-a-rag-agent.markdown %})

This post is kinda like an honest overview of my experiences, and what I've thought on engineering, machine learning (operations), through years. I think those questions were in your mind as well, and I'm not answering any of that, just sharing my opinions.

* Table of contents
{:toc}


### First ML problem, day-ahead electricity consumption forecasting

In 2021, I started working on energy consumption models, and it was my first real dive into operational applications. The problem was straightforward at first: we had to predict the daily electricity consumption for eight cities, for every 24 hours, but 24 hours ahead. Day-ahead electricity forecasting. Since I was working on the problem, the number of users kept increasing, the average consumer started using more energy, and new factories were being built. The demand for energy also shifted with changes in the economy. All means model drift and data drift.

After some time, I decided on the algorithms to use for forecasting models. After experimenting with deep learning models and tree-based models, I found that the most successful ones were LightGBM and XGBoost(and ensemble deep learning models).

The problem with tree-based models is that their predictions are often bounded by the data you provide. For example, if you try to forecast the first day of summer this year and last year's peak consumption was 600 unit, your prediction won’t exceed 600 units unless you formulate the problem correctly, due to the nature of tree-based models. However, when events and features are stable, and there isn’t much variation in output distributions, tree-based models can deliver excellent forecasts.

To be honest, most of my experience at the time was from Kaggle. I spent about nine months there, diving into forecasting competitions, creating new features, and trying different models. I came up with models that were 7/10, and I knew that if I were competing with real experts, they would probably tear my models apart.

### Realisation of the need for MLOps systems

Forecasting future energy consumption was a strange experience, especially because so many factors influence electricity usage—weather, weekends vs. weekdays, holidays, and even prayer times. Living in a place like Turkey, with its four distinct seasons, added another layer of complexity. I’m not a perfect model maker, and I struggled to make my models adapt to sudden changes. For instance, if the temperature drops 5°C in one day after two months of steady 25°C weather, the models often assumed that consumption would remain similar to the previous 25°C days. But that’s not how it works. People’s habits change depending on the season, and predicting those subtle shifts is tough. And then, there are the unpredictable events like national football matches—good luck modeling that!

So, what did I do to tackle model and data drift? I created hundreds of features over the past year, built tree-based models using XGBoost and LightGBM, and validated them over the last five days. MLFlow was a main driver of the logic for daily model production. While it might not have been the perfect way to validate models, it did improve long-term predictions. At least, it gave me a safe bet when consumptions regymes were stable, and the models adapted well.

To streamline the process, I set up an entire system where the data would be pulled through an API, features would be generated, models shortlisted, and predictions made every morning. It only took a minute to run, thanks to the automation I built. However, even with automation, you still had to keep an eye on the predictions, especially during national holidays or unexpected events.

### Although you deploy a model, you may not like it

One weird experience I had was an argument with my manager about predicting sudden drifts. He thought it would be easier to forecast lower consumption the next day if the weather dropped suddenly in the summer because fewer ACs would be running. He even manually adjusted the predictions a few times, and it backfired big time. That pretty much ended our debate. While I believed that some days could be more predictable with manual adjustments, the models often saw patterns we didn’t.

### Struggle of an ML guy in software engineering

Later, I transitioned to working on an MLOps platform in healthcare. I spent a lot of time looking for a job that would allow me to work on MLOps, and I was lucky to find a healthcare startup where I became the first full-time engineer. I had experience with healthcare models, so it felt like a perfect fit. It was pure luck how I found them, and they found me.

Working there for three years was a challenging process, especially because I moved from focusing on models to writing a platform. I’ve always been the type of a research guy, implement papers, and absorb as much as I could from Kaggle, professors, and academics. To illustrate, I was fascinated by the difference between deep learning models and tabular models, especially why deep learning often struggles with tabular data while tree-based models excel. I was reading papers, discussing the topic on twitter/reddit, i was that kind of guy. But when I switched to writing the platform, I realized how much I needed to learn about writing production-level code. I messed up a lot in the beginning.

In the first few months, we were running model evaluations in Jupyter notebooks, and I was reading papers on how to evaluate models in healthcare deeply, considering biases in gender and ethnicity. It was fun at first, but then we had to integrate everything into a platform. That’s when I slowly started to understand object-oriented programming, system design, and the principles of traditional software engineering. I'm non-CS engineer, so I learnt the things over time.

The startup life was intense, with long hours and a steep learning curve. The system we built used MongoDB, Python, RabbitMQ, S3, and AWS—a pretty standard pipeline. Our platform aimed to validate healthcare models, get approvals from the FDA, and ensure everything was done right. The data came from partners, but the model vendors never saw the raw data because they shouldn’t. So the business value proposition is we validate models on the data we obtain blind-foldedly and prepare the necessary documents for the FDA.

### MLOps Platform vs. Business Logic, do we only deploy model or serve for the customers?

For the platform to deliver on its promises, it needed to support all types of medical images, validate any computer vision model, and detect biases where they existed. Over the three years, the platform’s focus evolved. In the first year, we aimed to deploy and validate models. In the second year, we added annotation capabilities, supporting healthcare data along with cloud integration. By the third year, we realized we needed to focus on our customers' specific needs.

One of the challenges was separating the platform's logic from the client-specific codebase. We spent time writing client-specific code that would benefit 80% of the platform’s features, but there were times when we had to implement very specific logic for a particular client. This led to a lot of debates about whether we should enrich the platform with client-specific features or keep them separate. If you enrich the platform with too many client-specific features, it can become bloated and confusing. On the other hand, keeping them separate can lead to complications when the client codebase needs access to platform data.

I’m still unsure where to draw the line between MLOps, MLE, backend engineering, and business logic. There’s probably no single answer, but I think we did a great job maintaining and developing the platform.

### Deploying on Cloud vs. Deploying on-prem

Recently I've interviewed for MLOps role for banking, and I'm a guy that has written code %90 of the time. The manager that interviewed me was a DevOps-to-MLOps transitioned guy, I believe he has mildly developed ML models. For him, models are just dockers with specific output, and you need to manage/track/log. And for some of you guys, this is the real MLOps Engineering. I definitely kinda agree with that.

His team was deploying models on Apache Airflow, and he asked my experience on that. What I've realised was: 
- Models I trained/deployed on electricity problem was daily models as problem suggests, so I didn't have a need to track models but forecast. I didn't need to check if model is alive all the time, or a throughput, or a latency.
- Models I loaded to healthcare MLOps problems were project-based models, no need to constantly make the model up, and it was on-prem so security wasn't the main concern. 

His team's tradition was not writing a code at all, at least at the current time. It was a weird realisation for me during the interview.

### The Identity Crisis: MLOps Engineer, ML Engineer, or Both? Or others?

Another question that kept popping up in my head over the years was: who am I? Am I an MLOps engineer, an ML engineer, an ML researcher, or a backend engineer? In a small team or startup, you’re expected to be all of these things. You hear about being a "10x engineer," right? Those weird job posts that ask you to publish NeurIPS papers and code in Node.js simultaneously? Dude why the hell is that? I’ve seen a ton of these strange combinations. Not to brag, but here’s what I’ve done:

- Preprinted a paper from my Stanford internship = ML Researcher
- Written tabular and CV models = Data Scientist
- Implemented an Auto-ML module that allows instant training/fine-tuning of object detection, segmentation, and classification models, with versioning = Python Software Engineer, ML Engineering
- Developed an explainability library for CNN models using Grad-CAM = ML Researcher
- Serving models with FastAPI, integrations with frontend app = Backend Engineer
- Designed and implemented a platform using Docker, RabbitMQ, MongoDB, and other tools. = MLOps Engineer

Of course those terms are interchangeable, of course good MLE must be a good software engineer, of course you need to consider how to deploy models during training models, but what am I? I think I've fooled on "if I learn more, I'll gain more".

### The 10x Engineer Myth: Jack of All Trades, Master of Some(or None)

So, what did I gain from that? Am I a 10x engineer who isn’t perfect at everything? The upside is that I can apply for data science, MLOps, MLE, and Python backend roles with an ML focus. And I get interviewed for all of them. But what about the downside? During interviews, they don’t always believe in my breadth of experience. They ask tons of questions, and even if I have good answers, some interviewers try to trip me up. If you don’t give the specific answer they’re looking for, you’re out.

This happened to me in an interview with a global data science company. The guy questioned me on all the topics I mentioned, and switching contexts made me feel like an idiot. When he caught me giving a less-than-good answer, he said, “I knew it.” I mean, really, what did you know? Of course, I’m not an expert in everything. It was a weird flex. 

Another time, I was interviewed by a very intelligent guy—ex-well-known company, top 10 university PhD. Super wise, someone I’d like to be in ten years. He was kind, and I had total respect for him. I told him about my experience and what I’d done, and asked what more I could do. But guess what? He was looking for more experience in LLMs, and he said, "I don’t have the budget to invest in potential; I need experience in that specific topic." Fair enough, but what should I do then? Should I follow every deep learning trend and work on them commercially? How possible is that? What can you expect from someone with just four years of experience?

### Leetcodes on Interviews


I'll have tons of things to write on this later. 

### What do I wanna be in the future?

I don't know. I really don't know. What I know is: 
- FOMO is real in ML. 
- If you try to learn MLE, MLOps, Software Engineering and may be some DevOps won't make you more confident. 

