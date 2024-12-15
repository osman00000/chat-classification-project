# Task 0

I started by creating a sample conversatipn using ChatGPT.
Here I model, both the Customer and Sales Rep as separate agents with their own prompts.
Depending on the need these prompts might be expanded and can be templated with additional parameters. In this simple version, only input is the product name provided to the Customer system prompt. In order to allow a more natural end of conversation, I also add a dummy exit_conversation tool to the Customer calls after some turn. Additional tools can be added for more complex behaviors.

# Task 1
I created a simple wrapper around HuggingFace zero-shot-classifier to allow sentiment and intent classification. However, I tried to keep classification logic in the config file as much as possible. So, one can easily define a new classification dimension by adding a new item to PREDICTION_TYPES key in the config file. Here each prediction type is composed of labels and their textual representations. The textual representations are the text that go into the BART model. But since under the hood we may want to use longer pieces of texts here, I also defined a user friendly label name as keys in the json which are the labels seen by the user.

To run the code, first install the package :crossed_fingers:
```
pip install src/dist/chatclassifier-0.0.1.tar.gz
```
Then for an example, run 
```
python test.py
```
This should read the `config.ini` from `config` folder, pass it to the package which will download zero-shot model to `resources` folder and then process the conversation files in the `data` folder. The results will be both shown in the terminal and will be logged under the `log` folder.


# Task 2
To change this system to a conversational AI, without touching the sentiment/intent module, some of the potential "quick" things we could do:
* Once the product and intent is detected, we could retrieve relevant information pieces from product documentations (this needs to be indexed for fast retrieval). If our intent module is working well, that is giving us highly reliable keyword like outputs, we could maybe use a more filtered/lexical search, if not we could try a semantic search here or a hybrid between the two. Using product information together with some canned response forms ("here are relevant informations for you: {retrieved_product_info}. Let me know you need anything else"). The user experience will unnatural but with some randomization, one can also try to make the experience more natural.
* If we have a database of past conversations (between customer and human sales rep), we could categorize and index the pairs of questions and answers (QA-index). Then once we have an user input with clear intent, we could run a search with the intent and the question itself on the Q-part of the QA-index and return the best answer. Depending on our budget, a final reranking with a cross-encoder could increase the precision here.
* This might be more resource extensive compared the above ones but we could use small-LLMs to generate responses here. To improve the performance, we could use specialized intent-specific prompts.


# Task 3
* In my implentation I directly used pytorch code to serve the model which is not ideal. I think we can use ONNX runtime or NVidia Triton like optimized runtime engines. These compilers allow quantization, which can be helpful in decreasing the model size and inference times. 
* This might not be possible if we would like the run the code totally in-house but another potential deployment method is AWS Inferentia which sometimes provides some gains.
* We could use a smaller size distilled model or we can distil the model to a small size.
* Some optimization can be done with request batching which can increase overall throughput by decreasing overheads. However for some requests, this may sometimes require waiting for additional requests which can degrade the experience for some users.
* For very large models, one can distribute it across GPUs but for the case of BART I don't think this would be very meaningful as it is much smaller compared to more recent mdoels.
