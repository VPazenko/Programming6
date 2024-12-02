# Snakefile

# Configurations
import yaml

with open("config/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)


rule all:
    input:
        config['output_dir'] + "final_report.docx"


# Rule to make True Negative dataset
rule add_TN_data:
    input:
        config['data'] + config['eQTL_file_name']
    output:
        TN = config['data'] + 'True_Negative_df.tsv',
        full_df = config['data'] + 'full_df.tsv'
    script:
        "scripts/add_TN.py"


# Rule to preprocess the initial dataset (working/merging with different databases)
rule preprocess_data:
    input:
        lambda wildcards: list(chain(
            [config['data'] + 'full_df.tsv'],  # Первичный файл
            config.get(f'{wildcards.db}_db_name', [])  # Динамические файлы для конкретной базы данных
        ))
    output:
        config['inter_data'] + "{db}_output.csv"
    log:
        "logs/{db}_output.log"  # Логирование в файл
    params:
        script = lambda wildcards: wildcards.db + "_script.py"
    script:
        "scripts/{params.script}"




# Rule to combine all preprocesses data
rule combine_results_split:
    input:
        expand(config['inter_data'] + "{db}_output.csv", db=config['valuable_db'])  # Входные файлы всех стадий
    output:
        train = config['inter_data'] + 'train_data.csv', 
        test = config['inter_data'] + 'test_data.csv'
    script:
        "scripts/combine_split_test_train.py"


# Rule to train each model
rule train_model:
    input:
        config['inter_data'] + 'train_data.csv'
    output:
        model = config['output_dir'] + "models/{model}_model.pkl"
    params:
        model_name=lambda wildcards: wildcards.model
    benchmark: 
        config['output_dir'] + "benchmarks/train_{model}.txt"
    script:
        "scripts/train_{wildcards.model}.py"



# Rule to evaluate each model 'probably'
rule evaluate_model:
    input:
        model=config['output_dir'] + "models/{model}_model.pkl",
        data=config['inter_data'] + 'test_data.csv'
    output:
        results=temp(config['output_dir'] + "{model}_results.txt"),
        roc_curve=temp(config['output_dir'] + "{model}_roc_curve.png")
    params:
        model_name=lambda wildcards: wildcards.model
    script:
        "scripts/evaluate.py"


# Rule to create a final report from the results
rule create_report:
    input:
        results=expand(config['output_dir'] + "{model}_results.txt", model=config['models']),
        roc_curves=expand(config['output_dir'] + "{model}_roc_curve.png", model=config['models'])
    output:
        config['output_dir'] + "final_report.docx"
    script:
        "scripts/create_report.py"
