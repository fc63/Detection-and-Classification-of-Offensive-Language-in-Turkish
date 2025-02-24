This study aims to develop a model that detects whether a given turkish text contains toxic language.

Currently developed models are generally successful at detecting toxic language, but more data and fine-tuning are needed to categorize them.

Models trained as part of this study:
1. The model that i think is better than the others: https://huggingface.co/fc63/turkish-toxic-language-detection
2. The Model that i think fails to capture the context because there is too much lemmatization and stemming in the dataset: https://huggingface.co/fc63/toxic-category-model
3. The model that trained with the same processed dataset as the model in option 2, but tries to categorize as well as detect toxic language: https://huggingface.co/fc63/toxic-category-model
4. A lightweight model trained with Fasttext using the 3-gram n-gram method that detects only toxicity.
5. A lightweight model trained with Fasttext using the 3-gram n-gram method that both detects and categorizes toxicity.

Note: Fasttext models are not shared publicly, but the relevant code sections showing how they are trained are.

When training the model 1, the dataset was processed as little as possible, aiming to preserve the context. When training model 1, lemmatization and stemming were not performed on the dataset to preserve context. Instead, the zemberek grammar correction tool was used to convert the text in the lines that did not use a Turkish keyboard into turkish characters. The reason for this solution is that the bert transformer model recognizes the same word written in turkish characters and the same word written without turkish characters as different words. The reason for using this method only for lines without turkish characters is to keep the other unstructured data and make the model more suitable for informal language. Informal unstructured turkish is difficult to work with, a lot of data is needed to reduce false positives.

While training model 2, lemmatization was done with the Turkish library of stanza. Also, the words where the stanza returns none were stemmed manually. This method is effective in distinguishing different roots of words. In this way, we can make the model treat words with different affixes as the same word, but since the bert model is good at capturing the turkish context, we cannot use this advantage of the bert model in this method. So instead of lemmatizing and stemming, working with a larger data set and a good transformer helps us to get better results.Of course, if the dataset size is insufficient, it is better to do lemmatization and stemming.

When training Model 3, the same processed dataset as in Model 2 was used. Since there is more than one class, the softmax method was used. Especially for sexism and racism, the data in the dataset was insufficient. The Racism column correctly contains racist statements, but in return there are almost no non-racist rows containing race. This causes the model to label any mention of race as racist. If the shortcomings of the dataset are addressed by taking this problem into account, it will be more successful.

Another imbalance with low visibility in the dataset is that some words are used in toxic contexts much more than others. For example, the Turkish word “smart” has a good meaning on its own, but people tend to use it to use derogatory language against other people. When you do lemmatization and stemming, since the model is based on many more words, the word "zeki" is also marked as toxic regardless of context.

Another challenge when identifying toxic language is to identify ironic toxic language. Considering these factors, these models can be significantly improved through human supervision.
