# Evaluator

复用了FlashRAG评测生成答案质量的Evaluator，做到了即插即用。
支持metrics: rouge-1,rouge-2, rouge-l, bleu-1, bleu-2, bleu-3, bleu-4, f1, recall, precision。（其他的还没测）


数据集格式如下：  
generated response_file.jsonl
```
{“sample_id": $sample_id, "predicted_response": str}
```

