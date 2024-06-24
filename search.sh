curl -X POST \
  http://0.0.0.0:8003/find_matches \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is the role of GLP-1 and GLP-1 agonists in losing excess weight?",
    "k": 5
  }'
