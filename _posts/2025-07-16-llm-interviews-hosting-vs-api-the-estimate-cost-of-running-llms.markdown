---
layout: post
title:  "LLM Interviews : Hosting vs. API: The Estimate Cost of Running LLMs?"
date:   2025-05-16 00:00:10 +0300
categories: blog 
---

- I'll use any resource that helps me to prepare faster.
- I'm not directly working for interviews, but preparing for them makes me learn so many stuff, for years(check main page of website for 8 yo youtube channel).
- I'll cite every content I use. 
- I'll agressively reference anything I believe explained things better than me.


## Hosting vs. API: The True Cost of Running LLMs?

I was actually writing a new blog on LLM Interviews series, on prompt engineering. I realised the section below deserves its own blog.

There may be mistakes, contact me to fix it, I'll mention your name for anything helpful!

### How to estimate the cost of running SaaS-based and Open Source LLM models?

This is a very good question to distinguish people who deploys/decides to deploy LLM models. Our purpose as a team was to evaluate whether we should deploy model on GPUs (like [runpod](runpod.io), [Lambda Labs](https://lambdalabs.com), [Modal](https://modal.com), or [AWS Bedrock](https://aws.amazon.com), [Google Cloud Vertex](https://cloud.google.com), [Azure](https://azure.microsoft.com)) or use public SaaS-based APIs (such as [OpenAI](https://openai.com), [Anthropic](https://www.anthropic.com), [Cohere](https://cohere.com), or [Mistral API](https://mistral.ai)).

I don't do namedropping (branddropping) over here rather I'm trying to place names to use during interviews. I wish they hire me, I don't earn money thru this GitHub hosted website I swear :) . 


Let's go with some example.

Let's say we have an AI app, we estimate 100,000 api request in a month, on weekends demand/traffic is not high. We expect peak requests during afternoon. Our app reads 3-pages long pdf and returns a summary of it which is generally 250 words.

You should read [this](https://www.ptolemay.com/post/how-much-does-chatgpt-integration-cost-and-pay-off) 

#### 3.1 SaaS-based Models

API Pricings are mostly fixed to
- per model pricing (intelligent/bigger the model, cost is higher)
- Input/Output token based pricing (for each model, we have input/output token cost)

Let's go with [OpenAI pricing](https://openai.com/api/pricing/) (15 May 2025): 

| Model             | Description                                | Input (1M tokens) | Cached Input (1M tokens) | Output (1M tokens) |
|-------------------|--------------------------------------------|-------------------|--------------------------|---------------------|
| **GPT-4.1**       | Smartest model for complex tasks          | $2.00             | $0.50                    | $8.00               |
| **GPT-4.1 Mini**  | Affordable model balancing speed & intelligence | $0.40             | $0.10                    | $1.60               |
| **GPT-4.1 Nano**  | Fastest, most cost-effective model for low-latency tasks | $0.100            | $0.025                   | $0.400              |


We've experimented that our task is sufficient to be done with **GPT-4.1 Mini**, so we go with it.

Since scaling is not a deal for us, rather it's solved by OpenAI(still we will face problems). 


- A 3-page long PDF should be 500 * 3 = 1500 words.  
  - 1 word is approximately 0.75 tokens ([source](https://www.reddit.com/r/LocalLLaMA/comments/176u53g/is_the_075_tokens_per_word_rule_of_thumb_general/)).  
  - This means we'll have:  
    1500 words * 0.75 tokens/word = **1125 Input tokens** per request.

- On average, 250 words per output means:  
  250 words * 0.75 tokens/word = **187.5 Output tokens**.

- **Input Tokens**:  
  1125 tokens * 100,000 requests * $0.4 / 1 million tokens = **$45**  
- **Output Tokens**:  
  187.5 tokens * 100,000 requests * $1.60 / 1 million tokens = **$30**

**$45.00** (Input) + **$1.875** (Output) = **$75**

However, OpenAI and any API provider can have down time and they all have rate limits, so you should consider to add another API provider as a backup in case api request fails.

For example, in our apps we have hit to the 60 request per min limit frequently (`x-ratelimit-limit-requests`). In order to solve that:
- Either you can use batch processing, or delay the requests in codebase if response time is not critical
- Use another LLM api as a backup.


| FIELD                        | SAMPLE VALUE | DESCRIPTION                                                                 |
|------------------------------|--------------|-----------------------------------------------------------------------------|
| `x-ratelimit-limit-requests`  | 60           | Maximum number of requests allowed before exhausting the rate limit.        |
| `x-ratelimit-limit-tokens`    | 150000       | Maximum number of tokens allowed before exhausting the rate limit.          |
| `x-ratelimit-remaining-requests` | 59         | Remaining number of requests allowed before exhausting the rate limit.      |
| `x-ratelimit-remaining-tokens`   | 149984     | Remaining number of tokens allowed before exhausting the rate limit.        |
| `x-ratelimit-reset-requests`  | 1s           | Time remaining until the request rate limit resets.                          |
| `x-ratelimit-reset-tokens`    | 6m0s         | Time remaining until the token rate limit resets.                            |


But it's obvious $70 is very cheap than the deployment of LLM models.



#### 3.2 Open Source Model Deployment


Things get crazy in here. Let's discuss why you may need to choose Open Source models:

- 🛡️ In defense sector, companies don't want to use public API.  
- 🧮 You may use LLMs to categorize DB rows consistently/constantly, and you have millions of records.  
- 🧠 OpenAI may not fix your problem, you may want to fine-tune a better model.  
- 🌐 In remote locations or areas with limited connectivity, open-source models can be deployed locally, reducing reliance on the internet.  
- 📍 Data sovereignity, you may need to keep data in specific geographical regions to comply with legal or internal policies.


Let's assume you've found your model. And looking for a GPU to be used first.  

Before that, let's estimate our peak usage:

- 📈 100,000 api request in a month  
- 💤 On weekends demand/traffic is not high.  
- ☀️ We expect peak requests during afternoon.  

So:

- You assume in weekdays, you'll have %80 of the traffic (rather than 5 weekdays/7 days = 0.7)  
- You assume %50 of the traffic happens in the 4 hours of afternoon, and the rest 20 hours is similar.  
- `100,000 * 0.8 * 0.5 / ( 4 hours * 60 mins ) = 16 requests per minute` *on the average of peak hours*.  
- Let's quadruple the amount so that we know that in minutes that we have peak requests, we can still process.  
  → So in peak times we have **64 requests**.



You fine-tuned DeepSeek R1:8B and you seek for GPU that can inference.

- 🧪 You use DeepSeek R1:8B  
- 🔢 Q8 quantization is OK for you.  
- 📦 You want to inference one request (one batch size)

I've found an amazing website that does realistic calculations of VRAM/GPU needs per model, it's an amazing resource I discovered today. So you need to experiment over here:  
👉 [Can You Run This LLM? VRAM Calculator](https://apxml.com/tools/vram-calculator)


Let's say you choose RTX 4060 (16 GB) with 1 GPU.

- 🔄 On peak times, you have 64 requests.  
- 🧾 Max tokens per input, sequence length is 1125 inputs. Input size affects time-to-first-token speed.  
- ⚙️ Set number of GPUs to 1, we can increase if calculations show it's not enough.  
- 👥 Set concurrent users to 1, we will play with it to see if we can process peak demand.  

Website (kudos to them, those seem realistic with my experiments) says:  
**"Generation Speed: ~51 tok/sec per single user."**

However, we know that:

- ⏱️ On peak request, we expect 64 request per minute.  
- Which implies we need to end API calls at 0.93 seconds.  
- 🧠 We generate 187.5 Output tokens per request which takes 3.67 seconds to process single request.  
- 📉 Also with number of concurrent users, the token per second will still be lowered!


Let's assume we have found that with some methods mentioned below, we found we can fit our model and satisfy throughput constraints with 'RTX 5090', you've prepared your GGUF and deployed to Ollama/vLLM.

In Runpod.io, hourly cost of the pod is $0.89/hour. 

24 hours * 30 days * $0.89 / hour = $ 640.8 per month! Although it's bigger cost than the API, due to the various reasons we mentioned, it's bearable for corporations!


So this implies, the GPU is not enough! What to do now?

- 🪶 Try Q4 quantization or smaller if model quality doesn't drop!  
- 🖥️ Rent a bigger GPU, it seems 2x RTX 6000 gives ~188 tok/sec.  
- 🧩 Go with multi-GPU solution, but it'll have an engineering cost! I have never done that honestly :)  
- ⚗️ Fine-tune a smaller model, fine-tuning is not that hard to experiment!  
- 🤹 Go with batching on peak times, batch process 2 users, but it'll have an engineering cost! Again I have never done that honestly :)  
- 🌥️ If you still can use public API, use public API on peak times! A suitable GPU already handles most of the peak-ish demand, instead of trying to implement multi-GPU just use public API on overload times, you need to write backend of that.

Also beware:

- 💸 Deploying/training LLMs will steal your human power, salaries cost too!


---
Side Notes:
- %90 of the blog is/will be written by me.
- I may use ChatGPT to create %10 of the blogs, just to make content easier to read/to markdown format the algorithms. But nothing more than this. Just for help.
- This doesn't mean I create posts with ChatGPT, and proof-read and additions. It means I create posts by myself, in the middle I found ChatGPT can graph/format algorithm better than me, and prompt what I want to explain to it. Then it gives better format/fun to read content for just *some* part of the blog.
- The reason is, I already spend time to create content, and I'm learning while writing, but I understand content before writing, so I want to minimize writing time. 

