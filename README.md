# ELMo
An implementation of Embeddings from Language Models (ELMo). 


## Requirements
- python >= 3.10
- Other requirements listed in [requirements.txt](requirements.txt)

## Demo Usage in Downstream Tasks
### Sentiment Classification
- Run [train_forward_st.py](train_forward_st.py) and [train_backward_st.py](train_backward_st.py) to train the language models.
- This generates [forward_st.ckpt](forward_st.ckpt) and [backward_st.ckpt](backward_st.ckpt)
- Run [train_sentiment_classifier.py](train_sentiment_classifier.py) to train the sentiment classifier.
- This generates [sentiment_classifier.ckpt](sentiment_classifier.ckpt)

### Multi NLI
- Run [train_forward_multinli.py](train_forward_multinli.py) and [train_backward_multinli.py](train_backward_multinli.py) to train the language models.
- This generates [forward_multinli.ckpt](forward_multinli.ckpt) and [backward_multinli.ckpt](backward_multinli.ckpt)
- Run [train_multinli.py](train_multinli.py) to train the Multi NLI model.
- This generates [multinli_classifier.ckpt](multinli_classifier.ckpt)
