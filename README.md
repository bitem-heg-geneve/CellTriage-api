# Impaakt API

## Deployment
```
git clone https://github.com/bitem-heg-geneve/CellTriage-api.git

chmod 777 ./backend/app/run.sh

unzip ./backend/app/model_resources/impaakt.zip

docker compose -f "docker-compose.d`
```
## API 
This API allows to create literature triage jobs for the Cellosaurus knowledge resource on cell lines. Article relevancy scores are estimated using state-of-the art language models, fine-tuned for the triage of both abstracts and fulltext.

### Job Create endpoint
**URL:** POST [/api/job](/api/job)
Create a CellTriage job including a list of candicate articles. Each article must include a pmid. The articles will be scored for relevancy for Cellosaurus curation.

By default the system will attempt to extract the fulltext from PubMed Central. If this is not possible then the text will be exracted from the article abstract only. If the job -option "use_fulltext" is set false then the text will be retrieved from the abstract for all articles.

Example request body:
```
{
  "use_fulltext": true,
  "article_set": [
    {
      "pmid": 36585756
    },
    {
      "pmid": 36564873
    }
  ]
}
```


### Job Details endpoint
**URL:** GET [/api/job{job_id}]()

The article scores can be used to rank articles by estimated triage relavancy.
Text_source denotes if the text was retrieved from a the article abstract of from fulltext.

Example response
```
{
  "id": 0,
  "use_fulltext": true,
  "status": "pending",
  "job_created_at": "2023-09-08T19:38:03.193808",
  "process_start_at": "2023-09-08T20:26:01.333Z",
  "process_end_at": "2023-09-08T20:26:01.333Z",
  "process_time": 0,
  "article_set": [
    {
      "pmid": 0,
      "score": 0,
      "pmcid": "string",
      "text_source": "string",
      "text": "string"
    }
  ]
}
```
