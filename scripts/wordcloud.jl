using WordCloud, Random

Random.seed!(19)
words = ["Machine Learning", "Artificial Intelligence", "Neural Networks",
         "Statistics","Optimization", "Statistical Learning", "Data Science",
         "Loss Function", "Supervised Learning", "ML", "RMSE", "MLE", "Probability",
         "Unsupervised Learning", "Reinforcement Learning", "AI", "SGD", "ConvNet",
         "Classification", "Regression", "Clustering", "Random Forest", "LASSO",
         "Regularization", "Decision Tree", "Object Recognition", "Industry 4.0",
         "Bias-Variance Tradeoff", "Cross Validation", "Training Set", "Test Set",
         "Gradient Boosting", "PCA", "Logistic Regression", "kNN", "k-means",
         "Algorithms"]
weights = [3.2; 1.8; 1.5; 1.8; .8*rand(length(words)-4) .+ .6]
wc = wordcloud(words, weights,
               density = .35, nepoch = 10^3, colors = :seaborn_dark,
               angles = -45:15:45, transparentcolor = nothing,
               retry = 5, mask = shape(box, 800, 450)) |> generate!
paint(wc, "wc.png")
