"contemp poetry" is a Natural Language Processing project that aims to build classifiers distinguishing between good poems and bad poems.

Incorporating extra features improves the performance of most classifiers; (nested) cross-validation accuracy can be as high as 88%.

SentiWordNet is a lexical resource for opinion mining. It assigns to each WordNet synset three sentiment scores: positivity, negativity, and objectivity. One of our predictions is that good poets are better at making contrasts, and therefore having a higher sentiment-score fluctuation might be a sign of a good poem.

* contemppoetry.py - All the global names (constants, variables, and functions)  
* featextract.py - Using NLTK (Natural Language Toolkit) and SentiWordNet to extract extra features  
* classifier.py - Logistic Regression, Support Vector Machine, Decision Tree, K-Nearest Neighbors  
* featselect.py - Feature Selection--Recursive Feature Elimination  
* pca.py - Unsupervised Dimensionality Reduction--Principal Component Analysis  
* lda.py - Supervised Dimensionality Reduction--Linear Discriminant Analysis  
* kernelpca.py - Kernel Principal Component Analysis  
* ensemble.py - Ensemble Learning--Random Forest, Adaptive Boosting, Majority Vote Classifier  

This project is based on the research paper *A Computational Analysis of Style, Affect, and Imagery in Contemporary Poetry* by [Justine Kao](http://web.stanford.edu/~justinek/index.html) and [Dan Jurafsky](http://web.stanford.edu/~jurafsky/). We appreciate their generosity on providing the dataset.
