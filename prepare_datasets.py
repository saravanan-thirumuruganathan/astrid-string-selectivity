import xml.etree.ElementTree as ET
import re
import pandas as pd
import summary_data_structures

#Matches alphanumeric and space
regex_pattern = r'[^A-Za-z0-9 ]+'

#Download dblp50000.xml from HPI at
#https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/projekte/repeatability/DBLP/dblp50000.xml
def load_and_save_dblp():
    tree = ET.parse('dblp50000.xml')
    root = tree.getroot()

    titles = []
    authors = []

    for article in root:
        title = article.find("title").text
        titles.append(title)
        for authorNode in article.findall("author"):
            authors.append(authorNode.text)

    titles = pd.Series(titles)
    authors = pd.Series(authors)

    titles.str.replace(regex_pattern, '').to_csv("datasets/dblp/dblp_titles.csv", index=False, header=False)
    authors.str.replace(regex_pattern, '').to_csv("datasets/dblp/dblp_authors.csv", index=False, header=False)

#Download from https://github.com/gregrahn/join-order-benchmark
def load_imdb_movie_titles():
    df = pd.read_csv("title.csv", header=None, warn_bad_lines=True, error_bad_lines=False)
    #second column is the title
    df[1].str.replace(regex_pattern, '').to_csv("datasets/imdb/imdb_movie_titles.csv", index=False, header=False)

#Download from https://github.com/gregrahn/join-order-benchmark
def load_imdb_movie_actors():
    df = pd.read_csv("name.csv", header=None, warn_bad_lines=True, error_bad_lines=False)
    #second column is the title
    df[1].str.replace(regex_pattern, '').to_csv("datasets/imdb/imdb_movie_actors.csv", index=False, header=False)

#download from https://github.com/electrum/tpch-dbgen
#schema diagram at https://docs.deistercloud.com/content/Databases.30/TPCH%20Benchmark.90/Data%20generation%20tool.30.xml?embedded=true
def load_and_save_tpch():
    col_names = ["partkey", "name", "mfgr", "brand", "type", "size", "container", "retailprice", "comment"]
    df = pd.read_csv("part.tbl", sep='|', names = col_names,  warn_bad_lines=True, error_bad_lines=False, index_col=False)
    df["name"].str.replace(regex_pattern, '').to_csv("datasets/tpch/tpch_part_names.csv", index=False, header=False)

#This function will take a input file that contains strings one line at a time
#creates summary data structures for prefix, substring, suffix
#stores their selectivities
#and stores their triplets
#dataset_prefix is the name of the input file without .csv
#it will be used to generate outputs.
#eg. dblp_authors => dblp_authors.csv is the input file
#dblp_authors_prefix_count, dblp_authors_prefix_triplets contain the frequencies and triplets respectively
def prepare_dataset(folder_path, dataset_prefix):
    print("Processing ", dataset_prefix)
    functions = [summary_data_structures.get_all_prefixes, summary_data_structures.get_all_suffixes,
        summary_data_structures.get_all_substrings]
    function_desc = ["prefix", "suffix", "substring"]
    input_file_name = folder_path + dataset_prefix + ".csv"
    for index, fn in enumerate(functions):
        count_file_name = folder_path + dataset_prefix + "_" + function_desc[index] + "_counts.csv"
        triplet_file_name = folder_path + dataset_prefix + "_" + function_desc[index] + "_triplets.csv"


        tree = summary_data_structures.create_summary_datastructure(input_file_name, fn)
        #tree.print_tree()
        summary_data_structures.store_selectivities(tree, count_file_name)
        summary_data_structures.store_triplets(tree, triplet_file_name)


#The following functions generate the raw files for 4 datasets.
#Note: these are already in the github repository
#load_and_save_dblp()
#load_imdb_movie_titles()
#load_imdb_movie_actors()
#load_and_save_tpch()

#The following functions generates the frequencies and triplets
#This function might take few minutes for large datasets :)
if __name__ == "__main__":
    prepare_dataset("datasets/dblp/", "dblp_authors")
    prepare_dataset("datasets/dblp/", "dblp_titles")
    prepare_dataset("datasets/imdb/", "imdb_movie_actors")
    prepare_dataset("datasets/imdb/", "imdb_movie_titles")
    prepare_dataset("datasets/tpch/", "tpch_part_names")
