##Steps

First, please install the libraries from requirements.txt 

Overall, Astrid-Embed has 4 key steps.


1. Prepare necessary datasets. The code for processing the raw data into a format Astrid-Embed expects is in prepare_datasets.py. The functions load_and_save_dblp, load_imdb_movie_titles, load_and_save_tpch generates the files. For convenience, they are also put in the datasets folder.

2. Computing frequencies and triplets. The next step is to create the summary data structure for getting the prefix, suffix and substring frequencies. This is done by prepare_dataset function in prepare_datasets.py. You call it with two arguments dataset_path, dataset_prefix. For e.g. prepare_dataset("datasets/dblp/", "dblp_authors"). Astrid-Embed expects an input file with strings one string each line. So this function looks for dblp_authors.csv in the folder datasets/dblp/. It then generates 6 files. The files dblp_authors_prefix_counts.csv, dblp_authors_substring_counts.csv and dblp_authors_suffix_counts.csv are csv files that contains two columns : string, selectivity. So it stores the raw frequencies of prefix/suffix/substring for dblp_authors.csv. The other three files are dblp_authors_prefix_triplets.csv, dblp_authors_substring_triplets.csv, dblp_authors_suffix_triplets.csv. Each are csv files with three columns: Anchor,Positive,Negative.

3. You have to run the function prepare_dataset for the datasets. The commands are already provided in the prepare_datasets.py. The intermediate files are too large to share in Dropbox/GitHub. Except for IMDB which is very large, these intermediate files can be created in few seconds.

4. Learning Embedding Model and Selectivity Models: Astrid-Embed first learns an embedding model from the triplets and then uses a separate model to learn the selectivity. Both these training code are in AstrindEmbed.py. First, you have to alter the function setup_configs with the necessary configurations for training both these models such as the hyper parameters, input and output files. You can then call the main function to train embedding model, selectivity model and test it on a list of strings.

###Comments:
- AstridEmbed.py: The main function has some code to either train the model from scratch or reload from a previously saved model. For embedding model, this can be done by commenting out either train_astrid_embedding_model or load_embedding_model function. For selectivity estimator, the corresponding functions are train_selectivity_estimator and load_selectivity_estimation_model.
