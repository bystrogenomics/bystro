# Bystro API Documentation

## Signup

```sh
bystro-api signup --host https://bystro-dev.emory.edu --email alextest2@gmail.co
m --password testtest --name "Alex Foo"
```

<details>
<summary>Response</summary>

```json
Signing up for Bystro with email: alextest2@gmail.com, name: Alex Foo

Saved auth credentials to /home/alexkotlar/.bystro/bystro_authentication_token.json:
{
    "email": "alextest2@gmail.com",
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY2M2MwYWJmYTBlMTdhMTY2MGJhODdjNiIsIm5hbWUiOiJBbGV4IEZvbyIsInJvbGUiOiJ1c2VyIiwiZW1haWwiOiJhbGV4dGVzdDJAZ21haWwuY29tIiwiaWF0IjoxNzE1MjEwOTQzLCJleHAiOjE3MTUyMTA5NDR9.LMsjY8JWR7HXQwwaZJbdpW4jSm8jZo_tvijDyxNDse8",
    "url": "https://bystro-dev.emory.edu:443"
}

Signup & authentication successful. You may now use the Bystro API!
```

</details>

<br/>

## Get user profile

```sh
bystro-api get-user
```

<details>
<summary>Response</summary>

```json
{
  "options": {
    "autoUploadToS3": false
  },
  "_id": "663c0abfa0e17a1660ba87c6",
  "name": "Alex Foo",
  "email": "alextest2@gmail.com",
  "accounts": ["bystro"],
  "role": "user"
}
```

</details>

<br/>

## Create a new annotation

```sh
bystro-api create-annotation --files /home/alexkotlar/bystro/trio.trim.vep.vcf.bgz --assembly hg19
```

<details>
<summary>Response</summary>

```json
Job creation successful for /home/alexkotlar/bystro/trio.trim.vep.vcf.bgz
{
    "assembly": "hg19",
    "name": "trio_trim_vep_vcf_bgz-20240508235053964",
    "uploadProgress": {
        "resolved": true,
        "error": "",
        "progress": 1
    },
    "options": {
        "index": true
    },
    "inputFileNames": [
        "trio.trim.vep.vcf.bgz"
    ],
    "inputQueryConfig": {
        "fieldNames": [],
        "pipeline": []
    },
    "search": {
        "fieldNames": [],
        "archivedSubmissions": [],
        "queries": []
    },
    "type": "annotation",
    "visibility": "private",
    "_id": "663c0fdda0e17a1660ba880b",
    "archivedSubmissions": [],
    "expireDate": "2024-05-15T23:50:53.961Z",
    "userId": "663c0abfa0e17a1660ba87c6",
    "outputBaseFileName": "trio_trim_vep_vcf_bgz-20240508235053964",
    "submission": {
        "state": "submitted",
        "attempts": 0,
        "log": {
            "progress": 0,
            "skipped": 0,
            "messages": [
                "Job Submitted!"
            ]
        },
        "addedFileNames": [],
        "_id": "663c0fdea0e17a1660ba880c",
        "type": "annotation",
        "submittedDate": "2024-05-08T23:50:54.077Z",
        "queueId": "427068"
    },
    "createdAt": "2024-05-08T23:50:54.078Z",
    "updatedAt": "2024-05-08T23:50:54.085Z"
}
```

</details>

<br/>

## Get the job

```sh
bystro-api get-jobs --id 663c0fdda0e17a1660ba880b
```

<details>
<summary>Response</summary>

```json

Fetching job with id:   663c0fdda0e17a1660ba880b

Job(s) fetched successfully:

{
    "_id": "663c0fdda0e17a1660ba880b",
    "assembly": "hg19",
    "name": "trio_trim_vep_vcf_bgz-20240508235053964",
    "dirs": {
        "out": "/seqant/user-data/663c0abfa0e17a1660ba87c6/663c0fdda0e17a1660ba880b/output",
        "in": "/seqant/user-data/663c0abfa0e17a1660ba87c6/663c0fdda0e17a1660ba880b/input"
    },
    "uploadProgress": {
        "resolved": true,
        "error": "",
        "progress": 1
    },
    "options": {
        "index": true
    },
    "inputFileNames": [
        "trio.trim.vep.vcf.bgz"
    ],
    "inputQueryConfig": {
        "fieldNames": [],
        "pipeline": []
    },
    "search": {
        "fieldNames": [
            "chrom",
            "pos",
            "type",
            "inputRef",
            "alt",
            "trTv",
            "heterozygotes",
            "heterozygosity",
            "homozygotes",
            "homozygosity",
            "missingGenos",
            "missingness",
            "ac",
            "an",
            "sampleMaf",
            "vcfPos",
            "id",
            "discordant",
            "ref",
            "refSeq.siteType",
            "refSeq.exonicAlleleFunction",
            "refSeq.refCodon",
            "refSeq.altCodon",
            "refSeq.refAminoAcid",
            "refSeq.altAminoAcid",
            "refSeq.codonPosition",
            "refSeq.codonNumber",
            "refSeq.strand",
            "refSeq.name",
            "refSeq.name2",
            "refSeq.description",
            "refSeq.kgID",
            "refSeq.mRNA",
            "refSeq.spID",
            "refSeq.spDisplayID",
            "refSeq.protAcc",
            "refSeq.rfamAcc",
            "refSeq.tRnaName",
            "refSeq.ensemblID",
            "refSeq.isCanonical",
            "nearest.refSeq.name2",
            "nearest.refSeq.name",
            "nearest.refSeq.dist",
            "nearestTss.refSeq.name2",
            "nearestTss.refSeq.name",
            "nearestTss.refSeq.dist",
            "clinvarVcf.id",
            "clinvarVcf.alt",
            "clinvarVcf.AF_ESP",
            "clinvarVcf.AF_EXAC",
            "clinvarVcf.AF_TGP",
            "clinvarVcf.ALLELEID",
            "clinvarVcf.CLNDN",
            "clinvarVcf.CLNDNINCL",
            "clinvarVcf.CLNHGVS",
            "clinvarVcf.CLNREVSTAT",
            "clinvarVcf.CLNSIG",
            "clinvarVcf.CLNSIGCONF",
            "clinvarVcf.CLNVCSO",
            "clinvarVcf.DBVARID",
            "clinvarVcf.ORIGIN",
            "clinvarVcf.SSR",
            "clinvarVcf.RS",
            "gnomad.exomes.alt",
            "gnomad.exomes.id",
            "gnomad.exomes.AN",
            "gnomad.exomes.AF",
            "gnomad.exomes.AN_female",
            "gnomad.exomes.AF_female",
            "gnomad.exomes.non_cancer_AN",
            "gnomad.exomes.non_cancer_AF",
            "gnomad.exomes.non_neuro_AN",
            "gnomad.exomes.non_neuro_AF",
            "gnomad.exomes.non_topmed_AN",
            "gnomad.exomes.non_topmed_AF",
            "gnomad.exomes.controls_AN",
            "gnomad.exomes.controls_AF",
            "gnomad.exomes.AN_nfe_seu",
            "gnomad.exomes.AF_nfe_seu",
            "gnomad.exomes.AN_nfe_bgr",
            "gnomad.exomes.AF_nfe_bgr",
            "gnomad.exomes.AN_afr",
            "gnomad.exomes.AF_afr",
            "gnomad.exomes.AN_sas",
            "gnomad.exomes.AF_sas",
            "gnomad.exomes.AN_nfe_onf",
            "gnomad.exomes.AF_nfe_onf",
            "gnomad.exomes.AN_amr",
            "gnomad.exomes.AF_amr",
            "gnomad.exomes.AN_eas",
            "gnomad.exomes.AF_eas",
            "gnomad.exomes.AN_nfe_swe",
            "gnomad.exomes.AF_nfe_swe",
            "gnomad.exomes.AN_nfe_nwe",
            "gnomad.exomes.AF_nfe_nwe",
            "gnomad.exomes.AN_eas_jpn",
            "gnomad.exomes.AF_eas_jpn",
            "gnomad.exomes.AN_eas_kor",
            "gnomad.exomes.AF_eas_kor",
            "gnomad.genomes.alt",
            "gnomad.genomes.id",
            "gnomad.genomes.AN",
            "gnomad.genomes.AF",
            "gnomad.genomes.AN_female",
            "gnomad.genomes.AF_female",
            "gnomad.genomes.non_neuro_AN",
            "gnomad.genomes.non_neuro_AF",
            "gnomad.genomes.non_topmed_AN",
            "gnomad.genomes.non_topmed_AF",
            "gnomad.genomes.controls_AN",
            "gnomad.genomes.controls_AF",
            "gnomad.genomes.AN_nfe_seu",
            "gnomad.genomes.AF_nfe_seu",
            "gnomad.genomes.AN_afr",
            "gnomad.genomes.AF_afr",
            "gnomad.genomes.AN_nfe_onf",
            "gnomad.genomes.AF_nfe_onf",
            "gnomad.genomes.AN_amr",
            "gnomad.genomes.AF_amr",
            "gnomad.genomes.AN_eas",
            "gnomad.genomes.AF_eas",
            "gnomad.genomes.AN_nfe_nwe",
            "gnomad.genomes.AF_nfe_nwe",
            "gnomad.genomes.AN_nfe_est",
            "gnomad.genomes.AF_nfe_est",
            "gnomad.genomes.AN_nfe",
            "gnomad.genomes.AF_nfe",
            "gnomad.genomes.AN_fin",
            "gnomad.genomes.AF_fin",
            "gnomad.genomes.AN_asj",
            "gnomad.genomes.AF_asj",
            "gnomad.genomes.AN_oth",
            "gnomad.genomes.AF_oth",
            "dbSNP.id",
            "dbSNP.alt",
            "dbSNP.TOMMO",
            "dbSNP.ExAC",
            "dbSNP.GnomAD",
            "dbSNP.Korea1K",
            "dbSNP.GoNL",
            "dbSNP.KOREAN",
            "dbSNP.TWINSUK",
            "dbSNP.Vietnamese",
            "dbSNP.GENOME_DK",
            "dbSNP.GoESP",
            "dbSNP.GnomAD_exomes",
            "dbSNP.Siberian",
            "dbSNP.PRJEB37584",
            "dbSNP.SGDP_PRJ",
            "dbSNP.1000Genomes",
            "dbSNP.dbGaP_PopFreq",
            "dbSNP.NorthernSweden",
            "dbSNP.HapMap",
            "dbSNP.TOPMED",
            "dbSNP.ALSPAC",
            "dbSNP.Qatari",
            "dbSNP.MGP",
            "cadd",
            "caddIndel.alt",
            "caddIndel.PHRED"
        ],
        "archivedSubmissions": [],
        "queries": [],
        "activeSubmission": {
            "state": "completed",
            "attempts": 1,
            "log": {
                "progress": 13341,
                "skipped": 0,
                "messages": [
                    "Index Job Submitted!",
                    "Job Started!"
                ]
            },
            "addedFileNames": [],
            "_id": "663c0fe2a0e17a1660ba8817",
            "type": "searchIndex",
            "submittedDate": "2024-05-08T23:50:58.372Z",
            "queueId": "427075",
            "startedDate": "2024-05-08T23:50:59.380Z",
            "finishedDate": "2024-05-08T23:51:01.926Z"
        },
        "indexConfigPath": "hg19.mapping.yml"
    },
    "type": "annotation",
    "visibility": "private",
    "ancestry": {
        "submission": {
            "state": "completed",
            "attempts": 1,
            "log": {
                "progress": 0,
                "skipped": 0,
                "messages": [
                    "Job Submitted!",
                    "An ancestry worker has started your job."
                ]
            },
            "addedFileNames": [],
            "_id": "663c0fe2a0e17a1660ba881a",
            "submittedDate": "2024-05-08T23:50:58.382Z",
            "type": "ancestry",
            "queueId": "427076",
            "startedDate": "2024-05-08T23:50:59.388Z",
            "finishedDate": "2024-05-08T23:50:59.975Z"
        },
        "resultPath": "/seqant/user-data/663c0abfa0e17a1660ba87c6/663c0fdda0e17a1660ba880b/output/ancestry_results.json"
    },
    "proteomics": {},
    "archivedSubmissions": [],
    "expireDate": "2024-05-15T23:50:53.961Z",
    "userId": "663c0abfa0e17a1660ba87c6",
    "outputBaseFileName": "trio_trim_vep_vcf_bgz-20240508235053964",
    "submission": {
        "state": "completed",
        "attempts": 1,
        "log": {
            "progress": 13341,
            "skipped": 0,
            "messages": [
                "Job Submitted!",
                "An Amazon cloud worker has picked up your annotation.",
                "Checking input file format",
                "Beginning annotation",
                "Moving output files to output directory"
            ]
        },
        "addedFileNames": [],
        "_id": "663c0fdea0e17a1660ba880c",
        "type": "annotation",
        "submittedDate": "2024-05-08T23:50:54.077Z",
        "queueId": "427068",
        "startedDate": "2024-05-08T23:50:55.089Z",
        "finishedDate": "2024-05-08T23:50:58.356Z"
    },
    "createdAt": "2024-05-08T23:50:54.078Z",
    "updatedAt": "2024-05-08T23:51:01.927Z",
    "outputFileNames": {
        "statistics": {
            "json": "trio_trim_vep_vcf_bgz-20240508235053964.statistics.json",
            "tab": "trio_trim_vep_vcf_bgz-20240508235053964.statistics.tsv",
            "qc": "trio_trim_vep_vcf_bgz-20240508235053964.statistics.qc.tsv"
        },
        "sampleList": "trio_trim_vep_vcf_bgz-20240508235053964.sample_list",
        "config": "hg19.yml",
        "header": "trio_trim_vep_vcf_bgz-20240508235053964.annotation.header.json",
        "dosageMatrixOutPath": "trio_trim_vep_vcf_bgz-20240508235053964.dosage.feather",
        "log": "trio_trim_vep_vcf_bgz-20240508235053964.annotation.log.txt",
        "annotation": "trio_trim_vep_vcf_bgz-20240508235053964.annotation.tsv.gz"
    }
}
```

</details>

<br/>

## Query the job

Here we'll find all mutations that have the sample "4805" as a heterozygote or homozygote.

```sh
bystro-api query --job_id 663c0fdda0e17a1660ba880b --query "heteroyzogtes:4805 || homozygotes:4805" | jq
```

<details>

<summary>Response</summary>

```json
{
  "body": {
    "took": 2,
    "timed_out": false,
    "_shards": {
      "total": 1,
      "successful": 1,
      "skipped": 0,
      "failed": 0
    },
    "hits": {
      "total": {
        "value": 3997,
        "relation": "eq"
      },
      "max_score": 0.49784672,
      "hits": [
        {
          "_index": "663c0fdda0e17a1660ba880b_663c0abfa0e17a1660ba87c6",
          "_id": "763",
          "_score": 0.49784672,
          "_source": {
            "caddIndel": {
              "alt": [[[null]]],
              "PHRED": [[[null]]]
            },
            "trTv": [[[1]]],
            "heterozygosity": [[[0]]],
            "homozygotes": [[[1805], [1847], [4805]]],
            "refSeq": {
              "exonicAlleleFunction": [[["nonSynonymous"]]],
              "refAminoAcid": [[["R"]]],
              "kgID": [[[null], ["uc002gau.1", "uc002gav.1"]]],
              "spDisplayID": [[[null], ["UBP6_HUMAN"]]],
              "tRnaName": [[[null]]],
              "altAminoAcid": [[["Q"]]],
              "codonPosition": [[[2]]],
              "strand": [[["+"]]],
              "mRNA": [[[null], ["BX647719", "NM_004505"]]],
              "protAcc": [[["NP_001291213"], ["NP_004496"]]],
              "siteType": [[["exonic"]]],
              "codonNumber": [[[912]]],
              "name2": [[["USP6"]]],
              "spID": [[[null], ["P35125"]]],
              "rfamAcc": [[[null]]],
              "isCanonical": [[["false"], ["true"]]],
              "refCodon": [[["CGG"]]],
              "altCodon": [[["CAG"]]],
              "name": [[["NM_001304284"], ["NM_004505"]]],
              "description": [
                [
                  [null],
                  [
                    "Homo sapiens ubiquitin specific peptidase 6 (Tre-2 oncogene) (USP6), mRNA."
                  ]
                ]
              ],
              "ensemblID": [[[null], ["ENST00000250066", "ENST00000574788"]]]
            },
            "ref": [[["G"]]],
            "inputRef": [[["G"]]],
            "homozygosity": [[[1]]],
            "an": [[[6]]],
            "vcfPos": [[[5058808]]],
            "chrom": [[["chr17"]]],
            "nearestTss": {
              "refSeq": {
                "name2": [[["ZNF594"]]],
                "name": [[["NM_032530"]]],
                "dist": [[[24022]]]
              }
            },
            "dbSNP": {
              "ALSPAC": [[[0.659600019454956]]],
              "id": [[["rs9899177"]]],
              "Vietnamese": [[[0.148900002241135]]],
              "GENOME_DK": [[[0.75]]],
              "PRJEB37584": [[[null]]],
              "Siberian": [[[0.675000011920929]]],
              "1000Genomes": [[[0.337099999189377]]],
              "dbGaP_PopFreq": [[[0.607900023460388]]],
              "HapMap": [[[0.213799998164177]]],
              "ExAC": [[[0.508700013160706]]],
              "Korea1K": [[[0.10639999806881]]],
              "GoNL": [[[0.669300019741058]]],
              "TWINSUK": [[[0.642400026321411]]],
              "MGP": [[[0.670400023460388]]],
              "alt": [[["A"]]],
              "TOMMO": [[[0.111599996685982]]],
              "NorthernSweden": [[[0.618300020694733]]],
              "TOPMED": [[[0.498800009489059]]],
              "SGDP_PRJ": [[[0.670099973678589]]],
              "Qatari": [[[0.486099988222122]]],
              "GnomAD": [[[0.502600014209747]]],
              "KOREAN": [[[0.12049999833107]]],
              "GoESP": [[[0.52020001411438]]],
              "GnomAD_exomes": [[[0.51230001449585]]]
            },
            "type": [[["SNP"]]],
            "heterozygotes": [[[null]]],
            "missingness": [[[0]]],
            "missingGenos": [[[null]]],
            "ac": [[[6]]],
            "sampleMaf": [[[1]]],
            "clinvarVcf": {
              "CLNREVSTAT": [[[null]]],
              "CLNSIG": [[[null]]],
              "CLNVCSO": [[[null]]],
              "SSR": [[[null]]],
              "RS": [[[null]]],
              "alt": [[[null]]],
              "AF_EXAC": [[[null]]],
              "AF_TGP": [[[null]]],
              "CLNHGVS": [[[null]]],
              "CLNDNINCL": [[[null]]],
              "CLNSIGCONF": [[[null]]],
              "DBVARID": [[[null]]],
              "ORIGIN": [[[null]]],
              "id": [[[null]]],
              "AF_ESP": [[[null]]],
              "ALLELEID": [[[null]]],
              "CLNDN": [[[null]]]
            },
            "cadd": [[[23.1]]],
            "alt": [[["A"]]],
            "id": [[["."]]],
            "nearest": {
              "refSeq": {
                "name2": [[["USP6"]]],
                "name": [[["NM_001304284"], ["NM_004505"]]],
                "dist": [[[0]]]
              }
            },
            "gnomad": {
              "exomes": {
                "AN_nfe_seu": [[[11502]]],
                "AF_nfe_bgr": [[[0.651162981987]]],
                "AN_afr": [[[16230]]],
                "AN_sas": [[[30568]]],
                "AN_nfe_swe": [[[26128]]],
                "AN_female": [[[115394]]],
                "non_cancer_AN": [[[236618]]],
                "controls_AN": [[[109278]]],
                "AF_nfe_seu": [[[0.650583028793335]]],
                "AF_sas": [[[0.279900997877121]]],
                "AN_eas": [[[18374]]],
                "AF_eas_kor": [[[0.114053003489971]]],
                "AN": [[[251150]]],
                "AF_nfe_nwe": [[[0.649953007698059]]],
                "AF_eas_jpn": [[[0.0855263024568558]]],
                "non_neuro_AF": [[[0.501213014125824]]],
                "alt": [[["A"]]],
                "AN_amr": [[[34508]]],
                "AF": [[[0.512335002422333]]],
                "non_cancer_AF": [[[0.508249998092651]]],
                "AN_nfe_bgr": [[[2666]]],
                "AF_amr": [[[0.561869978904724]]],
                "AF_eas": [[[0.165777996182442]]],
                "AN_eas_kor": [[[3814]]],
                "AF_female": [[[0.514047980308533]]],
                "controls_AF": [[[0.493218988180161]]],
                "AN_nfe_onf": [[[30962]]],
                "non_topmed_AF": [[[0.515302002429962]]],
                "non_neuro_AN": [[[207834]]],
                "non_topmed_AN": [[[244542]]],
                "AF_afr": [[[0.261738002300262]]],
                "AF_nfe_onf": [[[0.64327198266983]]],
                "AF_nfe_swe": [[[0.648921012878418]]],
                "AN_nfe_nwe": [[[42120]]],
                "AN_eas_jpn": [[[152]]],
                "id": [[["rs9899177"]]]
              },
              "genomes": {
                "AF": [[[0.483904004096985]]],
                "controls_AN": [[[10848]]],
                "AN_afr": [[[8682]]],
                "AN_asj": [[[288]]],
                "alt": [[["A"]]],
                "AN_nfe_onf": [[[2134]]],
                "AF_nfe_onf": [[[0.636363983154297]]],
                "AN_fin": [[[3464]]],
                "AN": [[[31312]]],
                "AN_nfe_est": [[[4574]]],
                "id": [[["rs9899177"]]],
                "AN_female": [[[13884]]],
                "AF_fin": [[[0.46968799829483]]],
                "AF_nfe_seu": [[[0.650942981243134]]],
                "AF_afr": [[[0.268487006425858]]],
                "AN_amr": [[[846]]],
                "AN_eas": [[[1554]]],
                "AN_nfe": [[[15390]]],
                "AN_oth": [[[1088]]],
                "non_neuro_AN": [[[21204]]],
                "non_neuro_AF": [[[0.523155987262726]]],
                "AN_nfe_seu": [[[106]]],
                "AN_nfe_nwe": [[[8576]]],
                "AF_asj": [[[0.621528029441833]]],
                "AF_oth": [[[0.529411971569061]]],
                "AF_female": [[[0.489843994379044]]],
                "non_topmed_AF": [[[0.460447996854782]]],
                "controls_AF": [[[0.475755989551544]]],
                "AF_amr": [[[0.575649976730347]]],
                "AF_nfe_nwe": [[[0.641557991504669]]],
                "non_topmed_AN": [[[26522]]],
                "AF_eas": [[[0.176319003105164]]],
                "AF_nfe_est": [[[0.601005971431732]]],
                "AF_nfe": [[[0.628849983215332]]]
              }
            },
            "pos": [[[5058808]]],
            "discordant": [[["false"]]]
          }
        }
        //....
      ]
    }
  },
  "statusCode": 200,
  "headers": {
    "content-type": "application/json; charset=UTF-8",
    "content-length": "47973"
  },
  "meta": {
    "context": null,
    "request": {
      "params": {
        "method": "POST",
        "path": "/663c0fdda0e17a1660ba880b_663c0abfa0e17a1660ba87c6/_search",
        "body": {
          "type": "Buffer",
          "data": [
            31, 139, 8, 0, 0, 0, 0, 0, 0, 3, 61, 77, 203, 10, 194, 48, 16, 252,
            149, 178, 231, 32, 17, 91, 144, 220, 4, 207, 254, 66, 72, 113, 219,
            6, 211, 110, 221, 108, 14, 233, 227, 223, 77, 5, 157, 203, 48, 51,
            204, 204, 10, 29, 211, 8, 70, 43, 120, 39, 228, 12, 102, 133, 150,
            40, 28, 60, 166, 40, 7, 127, 3, 27, 133, 253, 212, 31, 250, 137,
            157, 75, 65, 44, 205, 200, 78, 136, 193, 192, 237, 113, 135, 255, 2,
            12, 40, 200, 148, 23, 234, 5, 163, 169, 175, 186, 169, 182, 173, 26,
            104, 164, 37, 247, 244, 243, 74, 33, 224, 228, 113, 42, 39, 194, 9,
            21, 204, 3, 187, 136, 54, 6, 154, 193, 52, 10, 196, 163, 109, 25,
            221, 11, 203, 135, 62, 93, 246, 2, 5, 209, 47, 8, 230, 172, 247, 15,
            180, 26, 44, 182, 188, 0, 0, 0
          ]
        },
        "querystring": "",
        "headers": {
          "user-agent": "opensearch-js/2.5.0 (linux 4.14.320-243.544.amzn2.x86_64-x64; Node.js v16.20.2)",
          "content-type": "application/json",
          "content-encoding": "gzip",
          "content-length": "162"
        },
        "timeout": 7200000
      },
      "options": {},
      "id": 2467
    },
    "name": "opensearch-js",
    "connection": {
      "url": "https://10.98.135.119:9200/",
      "id": "0HGqwOFiTdaeyg1cAjMRVg",
      "headers": {},
      "deadCount": 0,
      "resurrectTimeout": 0,
      "_openRequests": 0,
      "status": "alive",
      "roles": {
        "data": true,
        "ingest": true
      }
    },
    "attempts": 0,
    "aborted": false
  }
}
```

</details>
