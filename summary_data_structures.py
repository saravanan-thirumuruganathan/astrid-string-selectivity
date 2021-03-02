from collections import defaultdict
import pandas as pd
import random
from anytree import Node, NodeMixin, LevelOrderIter, RenderTree

#This is the maximum size of the prefix, suffix and substring that will be counted
MAX_STR_SIZE = 8


#Class to denote the node for a generic summary data structure.
class SummaryDSNode(NodeMixin):
    def __init__(self, name, parent=None, children=None):
        super(SummaryDSNode, self).__init__()
        self.name = name
        self.frequency = 1
        self.parent = parent
        self.char_to_children_dict = {}
        self.transition_probabilities = {}

    #Compute transition probabilities based on Eq 5 of the paper
    def update_transition_probabilities(self, root_node):
        k = len(self.children)
        total_frequency = sum([child.frequency for child in self.children])
        numerator, denominator = k , k+1

        if self.parent == root_node:
            numerator = k + 1
        else:
            self.transition_probabilities[self.parent] = 1.0 / denominator
        fraction = (numerator / denominator )
        for child in self.children:
            probability = 0.0
            if total_frequency > 0:
                probability = (child.frequency / total_frequency) * fraction
            self.transition_probabilities[child] = probability

#This class represents the entire generic summary data structure.
#Using a common for ease of coding.
#It can be replaced with more performant ones such as prefix trees, suffix trees etc.
class SummaryDataStructure:
    #string_generator_fn is a function that takes a string as input
    #and outputs a list of "substrings" of interest.
    #for e.g. all prefixes, suffixes,
    #max_str_size: will be the largest prefix, substring, suffix string that will be created
    #split_words: whether to ignore spaces in a string.
    #if split_words is true, then "a b" will be inserted as two words a b .. else one word with space.
    def __init__(self, string_generator_fn, max_str_size=MAX_STR_SIZE, split_words=True):
        self.string_generator_fn = string_generator_fn
        self.max_str_size = max_str_size
        self.split_words = split_words
        self.root_node = SummaryDSNode('')

    def insert_string(self, string):
        substrings_of_interest = self.string_generator_fn(string)
        for substring in substrings_of_interest:
            cur_node = self.root_node
            for index, char in enumerate(substring):
                if char in cur_node.char_to_children_dict:
                    cur_node = cur_node.char_to_children_dict[char]
                else:
                    new_node = SummaryDSNode(substring[:index+1], parent=cur_node)
                    cur_node.char_to_children_dict[char] = new_node
                    cur_node = new_node
            #Increment the frequency of the last node
            cur_node.frequency = cur_node.frequency + 1

    def update_summary_ds_from_file(self, input_file_name):
        with open(input_file_name) as f:
            for line in f:
                strings = [line.strip()]
                if self.split_words:
                    strings = line.strip().split()
                for string in strings:
                    self.insert_string(string)

    #returns a data frame with all the strings in the summary data structure and its frequencies
    def get_selectivities(self):
        string_frequency_dict = defaultdict(int)
        for node in LevelOrderIter(self.root_node):
            if node.is_root == False:
                string_frequency_dict[node.name] = max(1, node.frequency - 1)
        df = pd.DataFrame.from_dict(string_frequency_dict, orient='index')
        df.index.name = "string"
        df.columns = ["selectivity"]
        return df

    def update_transition_probabilities(self):
        for node in LevelOrderIter(self.root_node):
             if node.is_root == False:
                 node.update_transition_probabilities(self.root_node)

    #For each node,
    #   get the transition probabilities of going to other nodes
    #   use it to get num_triplets_per_node positive random samples using weighted sampling
    #   get num_triplets_per_node random strings as negative samples
    def get_triplets(self, random_seed=1234, num_triplets_per_node=4):
        random.seed(random_seed)
        self.update_transition_probabilities()

        #Get all the strings - it is needed to get dissimilar strings
        all_strings = [node.name for node in LevelOrderIter(self.root_node) if not node.is_root]
        total_nodes = len(all_strings)

        all_triplets = []

        for node in LevelOrderIter(self.root_node):
            #The root node is ornamental!
            if node.is_root:
                continue

            candidate_nodes = []
            candidate_probabilities = []

            #get all the neighbors of this node
            for other_node in node.transition_probabilities.keys():
                candidate_nodes.append(other_node.name)
                probability = node.transition_probabilities[other_node]
                candidate_probabilities.append(probability)

            for other_node in node.transition_probabilities.keys():
                for other_other_node in other_node.transition_probabilities.keys():
                    candidate_nodes.append(other_other_node.name)
                    #probability of reaching other_other_node from node
                    new_probability = probability * other_node.transition_probabilities[other_other_node]
                    candidate_probabilities.append(new_probability)


            if len(candidate_nodes) == 0:
                negatives = random.choices(population=all_strings, k=num_triplets_per_node)
                anchor = node.name
                for index in range(num_triplets_per_node):
                    all_triplets.append( (anchor, anchor, negatives[index]) )
                continue

            #normalize probabilities if needed
            candidate_probabilities_sum = sum(candidate_probabilities)
            candidate_probabilities = [elem/candidate_probabilities_sum for elem in candidate_probabilities]

            #Do a weighted random sampling of to get #num_triplets_per_node nodes
            # from candidates based num_triplets_per_node
            candidate_probabilities = list(candidate_probabilities)
            positives = random.choices(population=candidate_nodes, k=num_triplets_per_node, weights=candidate_probabilities)
            negatives = random.choices(population=all_strings, k=num_triplets_per_node)
            anchor = node.name
            for index in range(num_triplets_per_node):
                all_triplets.append( (anchor, positives[index], negatives[index]) )


        df = pd.DataFrame(all_triplets, columns = ["Anchor", "Positive", "Negative"])
        return df

    def print_tree(self):
        for pre, fill, node in RenderTree(self.root_node):
            print("%s%s:%d" % (pre, node.name, node.frequency))


def get_all_prefixes(string, max_size=MAX_STR_SIZE):
    return [string[:j] for j in range(1, min(max_size, len(string)) + 1)]

def get_all_suffixes(string, max_size=MAX_STR_SIZE):
    return [string[-j:] for j in range(1, min(max_size, len(string)) + 1)]

def get_all_substrings(string, max_size=MAX_STR_SIZE):
    arr = []
    n = len(string)
    for i in range(0,n):
        for j in range(i,n):
            if (j+1 - i) <= max_size:
                arr.append(string[i:(j+1)])
    return arr

#Naive way to compute all strings of interest that avoids the use of summary data structures
def aggregate_strings_of_interest(input_file_name, string_agg_fn,
    max_size=MAX_STR_SIZE, split_words=True, output_file_name=None):
    string_frequency_dict = defaultdict(int)

    with open(input_file_name) as f:
        for line in f:
            words = [line.strip()]
            if split_words:
                words = line.strip().split()
            for word in words:
                strings = string_agg_fn(word, max_size)
                for string in strings:
                    string_frequency_dict[string] += 1
    df = pd.DataFrame.from_dict(string_frequency_dict, orient='index')
    df.index.name = "string"
    df.columns = ["selectivity"]

    df = df.sort_values(by="string")
    if output_file_name is not None:
        df.to_csv(output_file_name, index=True, header=True)
    return df, string_frequency_dict


def create_summary_datastructure(input_file_name, string_generator_fn):
    tree = SummaryDataStructure(string_generator_fn)
    tree.update_summary_ds_from_file(input_file_name)
    return tree

def store_selectivities(tree, output_file_name):
    df = tree.get_selectivities()
    df = df.sort_values(by="string")
    df.to_csv(output_file_name, index=True, header=True)

def store_triplets(tree, output_file_name):
    df = tree.get_triplets()
    df.to_csv(output_file_name, index=False, header=True)
