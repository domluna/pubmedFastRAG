# pubmedFastRAG

This builds off of https://github.com/kyunghyuncho/pubmed-vectors by leveraging extremely fast binary exact search.

The original embeddings, which were ~110GB are compressed to 512 dimensions instead of 768 through [MRL](https://blog.nomic.ai/posts/nomic-embed-matryoshka). The 512 float32 values are then quantized to binary, resulting in 64 uint8 values. These values are then reinterpreted as 8 uint64 values which can be compared against each other with a single SIMD operation.

- `rag.jl` - This file contains the `RAGServer` which exposes an endpoint @ '0.0.0.0:8003/find_matches'. This endpoint accepts a POST request with a query (str) and k (int). `k` is the number of most relevant results to return.
- `embed.py` - This file contains a server that takes a query and returns a list of 64 uint8 values. This is called internally by `RAGServer`.

## Get Started

```sh
python embed.py
<All keys matched successfully>
INFO:     Started server process [888699]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)
```


```sh
# start a julia REPL with max threads
j --project=. -t auto
```

```julia
include("rag.jl")
# You need to create the database using the original repo linked above. It will be ~34GB.
rag = RAGServer("databases/pubmed_data.db")
start_server(rag)
```

You need the binary data for RAG. If you're interested in them message me.

```sh
λ ~/code/pubmedRAG: ls -lh bindata/
total 2.5G
-rw-rw-r-- 1 dom dom 2.2G Jun 23 13:22 data.bin
-rw-rw-r-- 1 dom dom 279M Jun 23 00:44 ids.bin
```

### Example query 1

> query = "What is the role of GLP-1 and GLP-1 agonists in losing excess weight?" k = 5

ID: 26961053, Distance: 86
Title: [GLP-1 agonist supports weight loss].
Authors: Maria Weiß
Publication Year: 2016
Abstract: Abstract not found...

ID: 37100640, Distance: 89
Title: Glucagon-like peptide 1 receptor agonists in end-staged kidney disease and kidney transplantation: A narrative review.
Authors: Kristin K Clemens, Jaclyn Ernst, Tayyab Khan, Sonja Reichert, Qasim Khan, Heather LaPier, Michael Chiu, Saverio Stranges, Gurleen Sahi, Fabio Castrillon-Ramirez, Louise Moist
Publication Year: 2023
Abstract: Glucagon-like peptide 1 receptor agonists (GLP-1RA) improve glycemic control and promote weight loss in type 2 diabetes (DM2) and obesity. We identified studies describing the metabolic benefits of GL...

ID: 36321278, Distance: 89
Title: Weight loss between glucagon-like peptide-1 receptor agonists and bariatric surgery in adults with obesity: A systematic review and meta-analysis.
Authors: Shohinee Sarma, Patricia Palcu
Publication Year: 2022
Abstract: Glucagon-like peptide-1 (GLP-1) receptor agonists recently demonstrated 15% to 20% weight loss in adults with obesity, a range which has previously been achieved only with bariatric surgery. This syst...

ID: 34160039, Distance: 90
Title: Glucagon-Like Peptide-1 (GLP-1) Receptor Agonism and Exercise: An Effective Strategy to Maintain Diet-Induced Weight Loss.
Authors: Leonarda Galiuto, Giovanna Liuzzo
Publication Year: 2021
Abstract: Abstract not found...

ID: 35914933, Distance: 90
Title: The role of GLP-1 receptor agonists in managing type 2 diabetes.
Authors: Noura Nachawi, Pratibha Pr Rao, Vinni Makin
Publication Year: 2022
Abstract: Glucagon-like peptide-1 (GLP-1) receptor agonists improve glycemic control in patients with type 2 diabetes mellitus, have cardioprotective and renoprotective effects, and do not cause weight gain or ...


### Example query 2
> query = "What are the biologies of TEAD?" k = 5

ID: 27421669, Distance: 116
Title: An evolutionary, structural and functional overview of the mammalian TEAD1 and TEAD2 transcription factors.
Authors: André Landin-Malt, Ataaillah Benhaddou, Alain Zider, Domenico Flagiello
Publication Year: 2016
Abstract: TEAD proteins constitute a family of highly conserved transcription factors, characterized by a DNA-binding domain called the TEA domain and a protein-binding domain that permits association with tran...

ID: 33611407, Distance: 116
Title: Exploring TEAD2 as a drug target for therapeutic intervention of cancer: A multi-computational case study.
Authors: Rajesh Pal, Amit Kumar, Gauri Misra
Publication Year: 2021
Abstract: Transcriptional enhanced associate domain (TEAD) is a family of transcription factors that plays a significant role during embryonic developmental processes, and its dysregulation is responsible for t...

ID: 36063664, Distance: 117
Title: A chemical perspective on the modulation of TEAD transcriptional activities: Recent progress, challenges, and opportunities.
Authors: Jianfeng Lou, Yuhang Lu, Jing Cheng, Feilong Zhou, Ziqin Yan, Daizhou Zhang, Xiangjing Meng, Yujun Zhao
Publication Year: 2022
Abstract: TEADs are transcription factors and core downstream components of the Hippo pathway. Mutations of the Hippo pathway and/or dysregulation of YAP/TAZ culminate in aberrant transcriptional activities of ...

ID: 28198677, Distance: 118
Title: Decipher the ancestry of the plant-specific LBD gene family.
Authors: Yimeng Kong, Peng Xu, Xinyun Jing, Longxian Chen, Laigeng Li, Xuan Li
Publication Year: 2017
Abstract: Lateral Organ Boundaries Domain (LBD) genes arise from charophyte algae and evolve essential functions in land plants in regulating organ development and secondary metabolism. Although diverse plant s...

ID: 33352993, Distance: 120
Title: Protein-Protein Interaction Disruptors of the YAP/TAZ-TEAD Transcriptional Complex.
Authors: Ajaybabu V Pobbati, Brian P Rubin
Publication Year: 2020
Abstract: The identification of protein-protein interaction disruptors (PPIDs) that disrupt the YAP/TAZ-TEAD interaction has gained considerable momentum. Several studies have shown that YAP/TAZ are no longer o...

## Timings

On my machine after JIT compilation here are the RAG timings (in seconds)

```
Time taken for RAG 0.09114909172058105
Time taken for DB query 0.0003159046173095703
Time taken for RAG 0.09531593322753906
Time taken for DB query 0.00039505958557128906
Time taken for RAG 0.08925509452819824
Time taken for DB query 0.04568600654602051
```

For embeddings (in seconds).

```
{'tokenization': 0.016788721084594727, 'model_inference': 0.0098114013671875, 'post_processing': 0.00022363662719726562, 'quantization': 2.5272369384765625e-05, 'total': 0.026870012283325195}
'total': 0.026870012283325195
```


Using CPU or GPU doesn't match a noticeable difference. The RAG timings do not differ either.
