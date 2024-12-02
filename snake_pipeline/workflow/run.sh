snakemake --snakefile workflow/main.smk 
snakemake --snakefile workflow/main.smk --dag | dot -Tpng > dag.png