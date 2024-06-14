import pandas as pd
import wget
import os
import tarfile
import glob
import pybtex.database
import bibtexparser
import edn_format


def get_param_file(model):
    path = "data/param_files/" 
    url_path = ("https://openkim.org/download/"
                f"{model}.txz"
                )
    file = wget.download(url_path , out= path)
    with tarfile.open(file) as tar:
        content = tar.extractall(f"{path}")
    os.remove(file)
    return


def get_param_files(models):
    """download model parameter files from OpenKIM
    """
    
    # pull params
    for i,model in enumerate(models):
        print(f"\n{i+1} of {(len(models))}: {model}")
        get_param_file(model)
    return 


def bib_file_cleanup():

    library = bibtexparser.parse_file("data/model_citations.bib")
    entry_keys = list(set(library.entries_dict.keys()))

    new_library = bibtexparser.Library()

    entry_keys_added = []
    for i,block in enumerate(library.blocks):
        if hasattr(block,"key"):
            if block.key in entry_keys_added:
                print(f"{block.key} already in")
            else:
                block.pop_field("month")
                new_library.add(block)
                entry_keys_added.append(block.key)

    bibtexparser.write_file("data/model_citations_new.bib", new_library)

    return


def create_citation_df(models_list, df_models):
    bib_ids = []
    for model in models_list:
        path_to_kimspec = f'./data/param_files/{model}/kimspec.edn'
        with open(path_to_kimspec,'r') as file:
            kimspec_text = file.read()
        kimspec = edn_format.loads(kimspec_text)
        source_list = []
        if 'source-citations' in kimspec.keys():
            for source in kimspec['source-citations']:
                source_list.append(f"OpenKIM-{source['recordkey'].replace('_',':')}") # source articles
            if 'extended-id' in kimspec.keys():
                source_list.append(f"OpenKIM-{kimspec['extended-id'].split('__')[-1].replace('_',':')}") # model
            if 'model-driver' in kimspec.keys():
                source_list.append(f"OpenKIM-{kimspec['model-driver'].split('__')[-1].replace('_',':')}") # model driver
        bib_ids.append(source_list)

    model_sources_key = {models_list[i]:bib_ids[i] for i in range(len(models_list))}

    return model_sources_key


def create_citation_file(df_models, model_sources_key):
    species_list = df_models.species.drop_duplicates()

    outfile = ""
    for species in species_list:
        models = df_models[df_models['species'] == species]['model'].to_list()
        outfile += species
        outfile += " \cite{"
        for model in models:
            sources = model_sources_key[model]
            if len(sources) > 0:
                for source in sources:
                    outfile += source
                    outfile += ", "
        outfile = outfile[:-2]
        outfile += "}, "
    outfile = outfile[:-2]
    return outfile

def create_citation_table(df_models, model_sources_key):
    species_list = df_models.species.drop_duplicates()

    citations = []
    for i,row in df_models.iterrows():
        current_model = row.model
        current_citations = model_sources_key[current_model]
        cite_list = ""
        for citation in current_citations:
            current = "\cite{" + citation + "},"
            cite_list += current
        cite_list = cite_list[:-1]
        citations.append(cite_list)

    df_models['citations'] = citations

    df_models = df_models.sort_values(['species','strength_MPa'])
    return df_models


def create_citation_list(df_models, model_sources_key):
    df_models = df_models.sort_values(['species','strength_MPa'])
    out_text = ""

    for i,row in df_models.iterrows():
        current_model = row.model
        current_citations = model_sources_key[current_model]
        current_string = f"{row.species}, {row.strength_MPa} "
        current_string += "\cite{"
        cite_list = ""
        for citation in current_citations:
            current = citation + ","
            cite_list += current
        cite_list = cite_list[:-1]
        current_string += cite_list + "}\\\\ \n"
        out_text += current_string
    return out_text

def main():
    # import list of models used
    df = pd.read_csv("data/models_w_props_full.csv")
    df_models = df[['species','model','strength_MPa']].drop_duplicates()
    models_list = df.model.drop_duplicates().to_list()
 
    if False:
        get_param_files(models_list)
    
    if False:
        # combine into one bib
        combined_bib = ""
        for model in models_list:
            kimcode = model.split("__")[1]
            with open(f'./data/param_files/{model}/kimcite-{kimcode}.bib','r') as file:
                lines = file.readlines()
                for line in lines:
                    combined_bib += line
                lines += "\n\n"

        # save to folder
        with open(f'./data/model_citations.bib','w') as out:
            out.write(combined_bib)

    
    # https://bibtexparser.readthedocs.io/en/main/quickstart.html#quickstart
    # https://kitchingroup.cheme.cmu.edu/blog/2014/02/10/Merging-bibtex-files-and-avoiding-duplicates/

    if False: # don't run after removing bib errors from file!
        bib_file_cleanup() 
    
    model_sources_key = create_citation_df(models_list, df_models)
    output_file = create_citation_file(df_models, model_sources_key)
    with open(f'./data/citation_text.txt','w') as out:
        out.write(output_file)
    
    output_df = create_citation_table(df_models, model_sources_key)
    output_df.to_csv("data/citations.csv")

    output_file = create_citation_list(df_models, model_sources_key)
    with open(f'./data/citation_list.txt','w') as out:
        out.write(output_file)

    return


if __name__ == "__main__":
    main()