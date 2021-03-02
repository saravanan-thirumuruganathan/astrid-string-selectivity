import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pandas as pd
import misc_utils
from string_dataset_helpers import TripletStringDataset, StringSelectivityDataset
import EmbeddingLearner
import SupervisedSelectivityEstimator

embedding_learner_configs, frequency_configs, selectivity_learner_configs = None, None, None


#This function gives a single place to change all the necessary configurations.
#Please see misc_utils for some additional descriptions of what these attributes mean
def setup_configs():
    global embedding_learner_configs, frequency_configs, selectivity_learner_configs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_learner_configs = misc_utils.AstridEmbedLearnerConfigs(embedding_dimension=64, batch_size=128,
        num_epochs=32, margin=0.2, device=device, lr=0.001, channel_size=8)

    path = "datasets/dblp/"
    #This assumes that prepare_dataset function was called to output the files.
    #If not, please change the file names appropriately
    file_name_prefix = "dblp_titles"
    query_type = "prefix"
    frequency_configs = misc_utils.StringFrequencyConfigs(
        string_list_file_name= path + file_name_prefix + ".csv",
        selectivity_file_name= path + file_name_prefix +  "_" +  query_type + "_counts.csv",
        triplets_file_name= path + file_name_prefix +  "_" +  query_type + "_triplets.csv"
    )

    selectivity_learner_configs = misc_utils.SelectivityEstimatorConfigs(
        embedding_dimension=64, batch_size=128, num_epochs=64, device=device, lr=0.001,
        #will be updated in train_selectivity_estimator
        min_val=0.0, max_val=1.0,
        embedding_model_file_name = path + file_name_prefix +  "_" +  query_type + "_embedding_model.pth",
        selectivity_model_file_name = path + file_name_prefix +  "_" +  query_type + "_selectivity_model.pth"
        )

    return embedding_learner_configs, frequency_configs, selectivity_learner_configs



#This function trains and returns the embedding model
def train_astrid_embedding_model(string_helper, model_output_file_name=None):
    global embedding_learner_configs, frequency_configs

    #Some times special strings such as nan or those that start with a number confuses Pandas
    df = pd.read_csv(frequency_configs.triplets_file_name)
    df["Anchor"] = df["Anchor"].astype(str)
    df["Positive"] = df["Positive"].astype(str)
    df["Negative"] = df["Negative"].astype(str)

    triplet_dataset = TripletStringDataset(df, string_helper)
    train_loader = DataLoader(triplet_dataset, batch_size=embedding_learner_configs.batch_size, shuffle=True)

    embedding_model = EmbeddingLearner.train_embedding_model(embedding_learner_configs, train_loader, string_helper)
    if model_output_file_name is not None:
        torch.save(embedding_model.state_dict(), model_output_file_name)
    return embedding_model

#This function performs min-max scaling over logarithmic data.
#Typically, the selectivities are very skewed.
#This transformation reduces the skew and makes it easier for DL to learn the models
def compute_normalized_selectivities(df):
    global selectivity_learner_configs
    normalized_selectivities, min_val, max_val = misc_utils.normalize_labels(df["selectivity"])
    df["normalized_selectivities"] = normalized_selectivities

    #namedtuple's are immutable - so replace them with new instances
    selectivity_learner_configs = selectivity_learner_configs._replace(min_val=min_val)
    selectivity_learner_configs = selectivity_learner_configs._replace(max_val=max_val)
    return df


#This function trains and returns the selectivity estimator.
def train_selectivity_estimator(train_df, string_helper, embedding_model, model_output_file_name=None):
    global selectivity_learner_configs, frequency_configs

    string_dataset = StringSelectivityDataset(train_df, string_helper, embedding_model)
    train_loader = DataLoader(string_dataset, batch_size=selectivity_learner_configs.batch_size, shuffle=True)

    selectivity_model = SupervisedSelectivityEstimator.train_selEst_model(selectivity_learner_configs, train_loader, string_helper)
    if model_output_file_name is not None:
        torch.save(selectivity_model.state_dict(), model_output_file_name)
    return selectivity_model

#This is a helper function to get selectivity estimates for an iterator of strings
def get_selectivity_for_strings(strings, embedding_model, selectivity_model, string_helper):
    global selectivity_learner_configs
    from SupervisedSelectivityEstimator import SelectivityEstimator
    embedding_model.eval()
    selectivity_model.eval()
    strings_as_tensors = []
    with torch.no_grad():
        for string in strings:
            string_as_tensor = string_helper.string_to_tensor(string)
            #By default embedding mode expects a tensor of [batch size x alphabet_size * max_string_length]
            #so create a "fake" dimension that converts the 2D matrix into a 3D tensor
            string_as_tensor = string_as_tensor.view(-1, *string_as_tensor.shape)
            strings_as_tensors.append(embedding_model(string_as_tensor).numpy())
        strings_as_tensors = np.concatenate(strings_as_tensors)
        #normalized_selectivities= between 0 to 1 after the min-max and log scaling.
        #denormalized_predictions are the frequencies between 0 to N
        normalized_predictions = selectivity_model(torch.tensor(strings_as_tensors))
        denormalized_predictions = misc_utils.unnormalize_torch(normalized_predictions, selectivity_learner_configs.min_val,
            selectivity_learner_configs.max_val)
        return normalized_predictions, denormalized_predictions


def load_embedding_model(model_file_name, string_helper):
    from EmbeddingLearner import EmbeddingCNNNetwork
    embedding_model= EmbeddingCNNNetwork(string_helper, embedding_learner_configs)
    embedding_model.load_state_dict(torch.load(model_file_name))
    return embedding_model

def load_selectivity_estimation_model(model_file_name, string_helper):
    from SupervisedSelectivityEstimator import SelectivityEstimator
    selectivity_model = SelectivityEstimator(string_helper, selectivity_learner_configs)
    selectivity_model.load_state_dict(torch.load(model_file_name))
    return selectivity_model

def main():
    random_seed = 1234
    misc_utils.initialize_random_seeds(random_seed)

    #Set the configs
    embedding_learner_configs, frequency_configs, selectivity_learner_configs = setup_configs()

    embedding_model_file_name = selectivity_learner_configs.embedding_model_file_name
    selectivity_model_file_name = selectivity_learner_configs.selectivity_model_file_name

    string_helper = misc_utils.setup_vocabulary(frequency_configs.string_list_file_name)

    #You can comment/uncomment the following lines based on whether you
    # want to train from scratch or just reload a previously trained embedding model.
    embedding_model = train_astrid_embedding_model(string_helper, embedding_model_file_name)
    #embedding_model = load_embedding_model(embedding_model_file_name, string_helper)

    #Load the input file and split into 50-50 train, test split
    df = pd.read_csv(frequency_configs.selectivity_file_name)
    #Some times strings that start with numbers or
    # special strings such as nan which confuses Pandas' type inference algorithm
    df["string"] = df["string"].astype(str)
    df = compute_normalized_selectivities(df)
    train_indices, test_indices = train_test_split(df.index, random_state=random_seed, test_size=0.5)
    train_df, test_df = df.iloc[train_indices], df.iloc[test_indices]

    #You can comment/uncomment the following lines based on whether you
    # want to train from scratch or just reload a previously trained embedding model.
    selectivity_model = train_selectivity_estimator(train_df, string_helper,
        embedding_model, selectivity_model_file_name)
    #selectivity_model = load_selectivity_estimation_model(selectivity_model_file_name, string_helper)

    #Get the predictions from the learned model and compute basic summary statistics
    normalized_predictions, denormalized_predictions = get_selectivity_for_strings(
        test_df["string"].values, embedding_model, selectivity_model, string_helper)
    actual = torch.tensor(test_df["normalized_selectivities"].values)
    test_q_error = misc_utils.compute_qerrors(normalized_predictions, actual,
        selectivity_learner_configs.min_val, selectivity_learner_configs.max_val)
    print("Test data: Mean q-error loss ", np.mean(test_q_error))
    print("Test data: Summary stats of Loss: Percentile: [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] ", [np.quantile(test_q_error, q) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]])


if __name__ == "__main__":
    main()
