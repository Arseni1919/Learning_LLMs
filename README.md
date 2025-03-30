# Learning LLMs (HuggingFace NLP Course)

## Terms


## Transformers

<img src="pics/tr_1.png" width="500">

There are a lot of papers that have a key impact of the field. Some of them are in my Mendeley library and will be covered here as well.

But, in general, all Transformer models can be categorised into three families of models:
- **GPT-like**: also called _auto-regressive_ Transf. models
- **BERT-like**: also called _auto-encoding_ Transf. models
- **BART/T5-like**: also called _sequence-to-sequence_ Transf. models

All models are trained in the self-superviased fasion: the objective is computed out of the input.
After that, there is a transfer learning - finetuning the model for a specific task.

### Auto-encoding Models 

The idea: take a text and make vector representation of the text. They trained by corrupting a given sentence, a random word in it, and asking the modfels with finding or reconstructing the initial sentence. The encoder (or auto-encoding) models use only the encoder of a Transformer model.

Usage example: sentence clasification, named entity recognition, extractive question answering (I give you a sentence and ask about the sentence. For example: Passage: "The Eiffel Tower was built in 1889 and is located in Paris, France." Question: "When was the Eiffel Tower built?")

Model Examples: ALBERT, BERT, DIstilBERT, ELECTRA, RoBERTa

### Auto-regressive Models

The idea: take the first words of the text (right shifted) and produce the next word (give a vector of probabilities for the next word). The pretraining here is to predict the next word in a sentence given previous words in the sentence. The decoder (or auto-regressive) models use only the decoder of a Transformer model.

Usage examples: text generation

Model Examples: CTRL, GPT, GPT-2, Transformer XL


### Sequence-to-Sequence Models


The idea: the encoder sees all the sentence, while decoder sees only the first part of the sentence. The pretraining is, for example, by replacing random spans of text (that can contain several words) with a single mask special word, and the objective is to predict those words. The encoder-decoder (or sequence-to-sequence) models use both parts of a Transformer model.

Usage examples: summarization, translation, generative question answering

Model Examples: BART, mBART, Marian, T5, mT5, Pegasus, ProphetNet, M2M100, MarianMT

Or it can be a combination of encoder + decoder models: BERT + GPT-2, BERT + BERT, RoBERTa + RoBERTa, etc.

In all of these models there will be always the intrinsic bias that will not dissappear.

### Example

Pipeline function:

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```


## HF Transformers

The pipeline function groups together 3 steps: preprocessing, passing the inputs through the model, and postprocessing:

<img src="pics/tr_2.png" width="500">

### Preprocessing with a tokenizer

Here,we use a tokenizer that: (1) splits the input to subwords / subsymbols, aka tokens; (2) maps each token to an integer; (3) adds additionla special tokens to the input.

An example: 

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```


### Going through the model

To download the model:

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

Model heads take the high-dimentional output and project it to a different dimention:

<img src="pics/tr_3.png" width="500">

In general you want to use something more specific to the task instead of `AutoModel`. Examples are: 
- Model (retrieve the hidden states)
- ForCausalLM
- ForMaskedLM
- ForMultipleChoice
- ForQuestionAnswering
- ForSequenceClassification
- ForTokenClassification
- and others 🤗

Example: 

```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```


### Postprocessing the output

To continue the exaple: 

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

Interpritation of the predictions:

```python
model.config.id2label
```


## Credits

Stand on the shoulders of giants.

- [HF | Learn](https://huggingface.co/learn)
- []()

