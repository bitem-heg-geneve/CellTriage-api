# Impaakt API

## Deployment
```
git clone https://github.com/bitem-heg-geneve/CellTriage-api.git

chmod 777 ./backend/app/run.sh

unzip ./backend/app/model_resources/impaakt.zip

docker compose -f "docker-compose.d`
```

## Search endpoint
This API allows to perform a fully customizable literature search for the [Cellosaurus](https://www.cellosaurus.org) knowledgebase.
The input is either a Lucene json query or a freetext query. The output is the Elasticsearc result set, ranked by relevance for Cellosaurus.

**URL:** [localhost:8001/api/search](http://localhost:8001/api/search)

**Mandatory input:** q OR jq: a query q in free text, which is interpreted by query analyzer, OR a Lucene json_query jq

**Optional input:** the number of requested documents (&n=, default 10, max 1000)

**Example:** simple search for MEDLINE (&col=) documents containing (&q=) Rhinolophus and Pangolin.

[sibils.text-analytics.ch/api/search?q=Rhinolophus%20and%20Pangolin&col=medline](http://localhost:8001/api/search?q=Rhinolophus%20and%20Pangolin)

Example: customizable search (&jq) with a Lucene style json query
```
{"query": {"bool" : {"must": {"match" : { "title" : "Digitoxin metabolism" }},"should" : {"match" : { "annotations_str" : "GO" }},"boost": 1}}}
```