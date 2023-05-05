Download Link: https://assignmentchef.com/product/solved-cse6242-homework-4-pagerank-algorithm-random-forest-scikit-learn
<br>
<strong>Q1 </strong>Implementation of Page Rank Algorithm

In this question, you will implement the PageRank algorithm in Python for large dataset.

The PageRank algorithm was first proposed to rank web search results, so that more “important” web pages are ranked higher.  It works by considering the number and “importance” of links pointing to a page, to estimate how important that page is. PageRank outputs a probability distribution over all web pages, representing the likelihood that a person randomly surfing the web (randomly clicking on links) would arrive at those pages.

As mentioned in the lectures, the PageRank values are the entries in the dominant eigenvector of the modified adjacency matrix in which each column’s values adds up to 1 (i.e., “column normalized”), and this eigenvector can be calculated by the power iteration method, which iterate through the graph’s edges multiple times to updating the nodes’ probabilities (‘scores’ in pagerank.py) in each iteration :

For each iteration, the Page rank computation for each node would be :

Where:

You will be using the dataset <a href="https://www.google.com/url?q=http://networkrepository.com/soc-wiki-elec.php&amp;sa=D&amp;ust=1574830382661000">Wikipedia adminship election data</a> which has almost 7K nodes and  100K edges. Also, you may find the dataset under the hw4-skeleton/Q1 as “soc-wiki-elec.edges”

In pagerank.py,

<ul>

 <li>You will be asked to implement the simplified PageRank algorithm, where<em> <strong>Pd ( vj ) = 1/n</strong></em> in the script provided and need to submit the output for 10, 25 iteration runs.</li>

</ul>

To verify, we are providing with the sample output of  5 iterations for simplified pagerank.

<ul>

 <li>For personalized PageRank, the <strong><em>Pd ( )</em></strong> vector will be assigned values based on your 9 digit GTID (Eg:</li>

</ul>

987654321) and you are asked to submit the output for 10, 25 iteration runs.

<strong>Deliverables:</strong>

<ol>

 <li><strong>py [12 pts]</strong>: your modified implementation</li>

 <li><strong>simplified_pagerank_{n}.txt</strong>: 2 files (as given below) containing the top 10 node IDs and their simplified pageranks for n iterations <strong>txt [2 pts] simplified_pagerank25.txt [2 pts]</strong></li>

 <li><strong>personalized_pagerank_{n}.txt: </strong>2 files (as given below) containing the top 10 node IDs and their simplified pageranks for n iterations <strong>txt [2 pts] personalized_pagerank25.txt [2 pts]</strong></li>

</ol>

<h1>Q2 Random Forest Classifier</h1>

<h2>Q2.1 – Random Forest Setup [45 pts]</h2>

<strong>Note: You must use Python 3.7.x for this question.</strong>

You will implement a random forest classifier in Python. The performance of the classifier will be evaluated via the out-of-bag (OOB) error estimate, using the provided dataset.

<strong>Note:</strong> You may only use the modules and libraries provided at the top of the .py files included in the skeleton for Q2 and modules from the Python Standard Library. Python wrappers (or modules) may NOT be used for this assignment. Pandas may NOT be used — while we understand that they are useful libraries to learn, completing this question is not critically dependent on their functionality. In addition, to make grading more manageable and to enable our TAs to provide better, more consistent support to our students, we have decided to restrict the libraries accordingly.




The dataset you will use is <a href="https://www.google.com/url?q=https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star&amp;sa=D&amp;ust=1574830382666000">Predicting a Pulsar Star</a> dataset. Each record consists of the parameters of a pulsar candidate. The dataset has been cleaned to remove missing attributes. The data is stored in a commaseparated file (csv) in your Q2 folder as <strong>pulsar_stars.csv. </strong>Each line describes an instance using 9 columns: the first 8 columns represent the attributes of the pulsar candidate, and the last column is the class which tells us if the candidate is a pulsar or not (1 means it is a pulsar, 0 means not a pulsar).

<strong>Note:</strong> The last column <strong>should not</strong> be treated as an attribute. <strong>Note2</strong>: Do not modify the dataset.

You will perform binary classification on the dataset to determine if a pulsar candidate is a pulsar or not.

<h3><strong>Essential Reading</strong></h3>

Decision Trees

To complete this question, you need to develop a good understanding of how decision trees work. We recommend you review the lecture on decision tree. Specifically, you need to know how to construct decision trees using <em>Entropy </em>and<em> Information Gain</em> to select the splitting attribute and split point for the selected attribute. These <a href="https://www.google.com/url?q=http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf&amp;sa=D&amp;ust=1574830382667000">slides from CMU</a> (also mentioned in lecture) provide an excellent example of how to construct a decision tree using <em>Entropy</em> and <em>Information Gain</em>.

Random Forests

To refresh your memory about random forests,  see Chapter 15 in the <a href="https://www.google.com/url?q=https://web.stanford.edu/~hastie/Papers/ESLII.pdf&amp;sa=D&amp;ust=1574830382667000">Elements of Statistical Learning</a> book and the lecture on random forests. Here is a <a href="https://www.google.com/url?q=http://blog.echen.me/2011/03/14/laymans-introduction-to-random-forests/&amp;sa=D&amp;ust=1574830382668000">blog post</a> that introduces random forests in a fun way, in layman’s terms.

<h4>Out-of-Bag Error Estimate</h4>

<strong>In random forests, it is not necessary to perform explicit cross-validation or use a separate test set for performance evaluation. Out-of-bag (OOB) error estimate has shown to be reasonably accurate and </strong><a href="https://www.google.com/url?q=https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm%23ooberr&amp;sa=D&amp;ust=1574830382668000"><strong>unbiased. Below, we summarize the key points about OOB described in the</strong></a> <a href="https://www.google.com/url?q=https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm%23ooberr&amp;sa=D&amp;ust=1574830382668000"><strong>original article by Breiman and Cutler</strong></a><a href="https://www.google.com/url?q=https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm%23ooberr&amp;sa=D&amp;ust=1574830382668000"><strong>.</strong></a>

Each tree in the forest is constructed using a different bootstrap sample from the original data. Each bootstrap sample is constructed by randomly sampling from the original dataset <strong>with replacement </strong>(usually, a bootstrap sample has the <a href="https://www.google.com/url?q=http://stats.stackexchange.com/questions/24330/is-there-a-formula-or-rule-for-determining-the-correct-sampsize-for-a-randomfore&amp;sa=D&amp;ust=1574830382669000">same size</a> as the original dataset). Statistically, about one-third of the cases are left out of the bootstrap sample and not used in the construction of the <em>kth</em> tree. For each record left out in the construction of the <em>kth</em> tree, it can be assigned a class by the <em>kth</em> tree. As a result, each record will have a “test set” classification by the subset of trees that treat the record as an out-of-bag sample. The majority vote for that record will be its predicted class. The proportion of times that a predicted class is not equal to the true class of a record averaged over all records is the OOB error estimate.

<h3><strong>Starter Code</strong></h3>

We have prepared starter code written in Python for you to use. This would help you load the data and evaluate your model. The following files are provided for you:

<ul>

 <li>py: utility functions that will help you build a decision tree</li>

 <li>py: a decision tree class that you will use to build your random forest</li>

 <li>py: a random forest class and a main method to test your random forest</li>

</ul>

<h3><strong>What you will implement</strong></h3>

Below, we have summarized what you will implement to solve this question. Note that you MUST use <strong>information gain</strong> to perform the splitting in the decision tree. The starter code has detailed comments on how to implement each function.

<ol>

 <li>py: implement the functions to compute entropy, information gain, and perform splitting.</li>

 <li>py: implement the learn() method to build your decision tree using the utility functions above.</li>

 <li>py: implement the classify() method to predict the label of a test record using your decision tree.</li>

 <li>py: implement the methods _bootstrapping(), fitting(), voting() <strong>Note</strong>: You must achieve a minimum accuracy of 75% for the random forest.</li>

</ol>

<strong>Note 2</strong>: Your code must take no more than 5 minutes to execute.

<strong>Note 3</strong>: Remember to remove all of your print statements from the code. Nothing other than the existing print statements in <strong>main()</strong> should be printed on the console. Failure to do so may result in point deduction. Do not remove the existing print statements in <strong>main()</strong> in <strong>random_forest.py</strong>.

As you solve this question, you will need to think about multiple parameters in your design, some may be more straightforward to determine, some may be not (hint: study lecture slides and essential reading above). For example:

<ul>

 <li>Which attributes to use when building a tree?</li>

 <li>How to determine the split point for an attribute?</li>

 <li>When do you stop splitting leaf nodes?</li>

 <li>How many trees should the forest contain?</li>

</ul>

Note that, as mentioned in lecture, there are other approaches to implement random forests. For example, <a href="https://www.google.com/url?q=http://citeseerx.ist.psu.edu/viewdoc/download?doi%3D10.1.1.232.2940%26rep%3Drep1%26type%3Dpdf&amp;sa=D&amp;ust=1574830382672000">instead of information gain, other popular choices include Gini index, random attribute selection (e.g., </a><a href="https://www.google.com/url?q=http://citeseerx.ist.psu.edu/viewdoc/download?doi%3D10.1.1.232.2940%26rep%3Drep1%26type%3Dpdf&amp;sa=D&amp;ust=1574830382672000">PERT Perfect Random Tree Ensembles</a><a href="https://www.google.com/url?q=http://citeseerx.ist.psu.edu/viewdoc/download?doi%3D10.1.1.232.2940%26rep%3Drep1%26type%3Dpdf&amp;sa=D&amp;ust=1574830382672000">). We decided to ask everyone to use an information gain based approach in </a>this question (instead of leaving it open-ended), to help standardize students’ solutions to help accelerate our grading efforts.

<h2>Q2.2 – forest.txt</h2>

In <strong>forest.txt</strong>, report the following:

<ol>

 <li>What is the main reason to use a random forest versus a decision tree? (&lt;= 50 words)</li>

 <li>How long did your random forest take to run? (in seconds)</li>

 <li>What accuracy (to two decimal places, xx.xx%) were you able to obtain?</li>

</ol>

<h3><strong>Deliverables</strong></h3>

<ol>

 <li><strong>py</strong>: The source code of your utility functions.</li>

 <li><strong>py</strong>: The source code of your decision tree implementation.</li>

 <li><strong>py </strong>: The source code of your random forest implementation with appropriate comments.</li>

 <li><strong>txt </strong>: The text file containing your responses to Q2.2</li>

</ol>

<h2>Q3  Using Scikit-Learn</h2>

<strong>Note: You must use Python 3.7.x for this question.</strong>

<a href="https://www.google.com/url?q=http://scikit-learn.org&amp;sa=D&amp;ust=1574830382675000">Scikit-learn</a> is a popular Python library for machine learning. You will use it to train some classifiers on the <em>Predicting a Pulsar Star</em><sup>[<u>1</u>]</sup> dataset which is provided in the hw4-skeleton/Q3 as <em>pulsar_star.csv</em>

<strong>Note</strong>: Your code must take no more than 15 minutes to execute all cells.

—————————————————————————————————————————-

<strong>For this problem you will be utilizing and submitting a </strong><a href="https://www.google.com/url?q=https://jupyter.readthedocs.io/en/latest/install.html&amp;sa=D&amp;ust=1574830382676000"><strong>Jupyter notebook</strong></a><strong>.</strong>

For any values we ask you to report in this question, please make sure to print them out in your Jupyter notebook such that they are outputted when we run your code.

<strong>NOTE: DO NOT ADD ANY ADDITIONAL CELLS TO THE JUPYTER NOTEBOOK AS IT WILL CAUSE THE AUTOGRADER TO FAIL.</strong>

<strong>NOTE: The below instructions will not match the exact flow of the Juypter notebook, you will have to find the section that applies to each of the different classifiers.</strong>

—————————————————————————————————————————-

<h3>Q3.1 – Classifier Setup</h3>

Train each of the following classifiers on the dataset, using the classes provided in the links below. You will do hyperparameter tuning in Q3.2 to get the best accuracy for each classifier on the dataset.

<ol>

 <li><a href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html%23sklearn.linear_model.LinearRegression&amp;sa=D&amp;ust=1574830382677000">Linear Regression</a></li>

 <li><a href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html&amp;sa=D&amp;ust=1574830382678000">Random Forest</a></li>

 <li><a href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html%23sklearn.svm.SVC&amp;sa=D&amp;ust=1574830382678000">Support Vector Machine</a> (The link points to SVC, which is a particular implementation of SVM by Scikit.)</li>

 <li><a href="https://www.google.com/url?q=https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html&amp;sa=D&amp;ust=1574830382679000">Principal Component Analysis</a></li>

</ol>

Scikit has additional documentation on each of these classes, explaining them in more detail, such as how they work and how to use them.

Use the jupyter notebook skeleton file called <strong>hw4q3.ipynb</strong> to write and execute your code.

As a reminder, the general flow of your machine learning code will look like: 1. Load dataset

<ol start="2">

 <li>Preprocess (you will do this in Q3.2)</li>

 <li>Split the data into x_train, y_train, x_test, y_test</li>

 <li>Train the classifier on x_train and y_train</li>

 <li>Predict on x_test</li>

 <li>Evaluate testing accuracy by comparing the predictions from step 5 with y_test.</li>

</ol>

Here is an <a href="https://www.google.com/url?q=http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html%23sphx-glr-auto-examples-linear-model-plot-ols-py&amp;sa=D&amp;ust=1574830382682000">example</a><a href="https://www.google.com/url?q=http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html%23sphx-glr-auto-examples-linear-model-plot-ols-py&amp;sa=D&amp;ust=1574830382682000">.</a> Scikit has many other examples as well that you can learn from.

<h3>Q3.2 – Hyperparameter Tuning</h3>

Tune your random forest and SVM to obtain their best accuracies on the dataset. For random forest, tune the model on the unmodified test and train datasets. For SVM, either <a href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html&amp;sa=D&amp;ust=1574830382683000">standardize</a> or <a href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html&amp;sa=D&amp;ust=1574830382683000">normalize</a> the dataset before using it to tune the model.

<strong>Note:</strong>

If you are using StandardScaler:

<ul>

 <li>Pass x_train into the fit method. Then transform both x_train and x_test to obtain the standardized versions of both.</li>

 <li>The reason we fit only on x_train and not the entire dataset is because we do not want to train on data that was affected by the testing set.</li>

</ul>

Tune the hyperparameters specified below, using the <a href="https://www.google.com/url?q=http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html&amp;sa=D&amp;ust=1574830382685000">GridSearchCV</a> function that Scikit provides:

<ul>

 <li>For random forest, tune the parameters “n_estimators” and “max_depth”.</li>

 <li>For SVM, tune “C” and “kernel” (try only ‘linear’ and ‘rbf’).</li>

</ul>

Use <strong>10 folds</strong> by setting the <em>cv</em> parameter to 10.

You should test at least 3 values for each of the numerical parameters. For C, the values should be different by factors of at least 10, for example, 0.001, 0.01, and 0.1, or 0.0001, 0.1 and 100.

<strong>Note: </strong>If GridSearchCV is taking a long time to run for SVM, make sure you are standardizing or normalizing your data beforehand.

<h3>Q3.3 – Cross-Validation Results</h3>

Let’s practice getting the results of cross-validation. For your SVM (only), report the <em>rank test score, mean testing score </em>and <em>mean fit time</em> for the best combination of hyperparameter values that you obtained in Q3.2. The GridSearchCV class holds a  ‘cv_results_’ dictionary that should help you report these metrics easily.

<h3>Q3.4 – Evaluate the relative importance of features</h3>

You have performed a simple classification task using the random forest algorithm. You have also implemented the algorithm in Q2 above. The concept of entropy gain can also be used to evaluate the importance of a feature.

In this section you will determine the feature importance as evaluated by the random forest Classifier. You must then sort them in descending order and print the feature numbers. Hint: There is a direct function available in sklearn to achieve this. Also checkout argsort() function in Python numpy.

(argsort() returns the indices of the elements in ascending order)

You should use the first classifier that you trained initially in <strong>Q3.1</strong>, without any kind of hyperparametertuning, for reporting these features.

.

<h3>Q3.5 – Principal Component Analysis [2 pts]</h3>

Dimensionality reduction is an important task in many data analysis exercises and it involves projecting the data to a lower dimensional space using Singular Value Decomposition. Refer to the examples given <a href="https://www.google.com/url?q=https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html&amp;sa=D&amp;ust=1574830382689000">here</a><a href="https://www.google.com/url?q=https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html&amp;sa=D&amp;ust=1574830382689000">,</a> set parameters <em>n_component</em> to 8 and <em>svd_solver</em> to ‘full’ (keep other parameters at their default value), and report the following in the relevant section of hw4q3.ipynb:

<ol>

 <li><strong><u>Percentage</u></strong> of variance explained by each of the selected components. Sample Output:</li>

</ol>

[6.51153033e-01 5.21914311e-02 2.11562330e-02 5.15967655e-03

6.23717966e-03 4.43578490e-04 9.77570944e-05 7.87968645e-06]

<ol start="2">

 <li>The singular values corresponding to each of the selected components.</li>

</ol>

Sample Output:

[5673.123456  4532.123456   4321.68022725  1500.47665361    1250.123456   750.123456    100.123456    30.123456]

Deliverables

– <strong>hw4q3.ipynb </strong>– jupyter notebook file filled with your code for part Q3.1-Q3.5.