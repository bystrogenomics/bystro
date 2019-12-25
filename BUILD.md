# Configuring Bystro for building new databases

## Utilities
Bystro comes with several small programs to fetch and modify source files

## 'Fetch'

TL;DR : Fetches from some remote repository or SQL server. Provide the remote file path or SQL statement, and Bystro will fetch the data.

### 1. Using SQL statements

###### Synopsis:
```yaml
  type: gene
  utils:
  - name: fetch
    args:
      sql_statement: SELECT * FROM refGene r LEFT JOIN kgXref k ON r.name = k.refseq WHERE chrom = %chromosomes%;
      connection:
        host: genome-mysql.cse.ucsc.edu
        user: 'genome'
        database: 'hg19'
  - args:
      geneFile: /mnt/db_backup/dbnsfp/dbNSFP3.5_gene.complete
    completed: 2017-11-23T19:25:00
    name: refGeneXdbnsfp
```

Any valid SQL statement can be used. In order to pre-split the fetched data per chromosome, simply provide the `%chromosomes%` macro (ex: `WHERE chrom = %chromosomes%`)

###### Optional parameters:
- All `args` except `sql` are optional for SQL fetch. The following options will be assumed:
```yaml
  args:
    connection:
      host: genome-mysql.cse.ucsc.edu
      user: 'genome'
      database: //The `assembly` (ex: hg19)
```