# Prompt Injection attack Guard (InjectGuard)


LLM has been widely applied in many fields. However, some `prompt injection attacks (PIA)`, such as `jailbreak attack`, `hijacking attack`, `prompt leakage`, etc., cause LLM to have security risks. Some AI safety systems highlight the serious risks that PIA may pose, such as [MITRE ATLAS](https://atlas.mitre.org/) and [OWASP TOP 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
***

## Motivation 💡
* **Investigate** the latest injection attack detection methods.

* **Open-source** various types of injection attack detection solutions.

* **Benchmark** the effectiveness of injection attack detection methods across different languages and scenarios.


## Newest Solution 👾
| Company    | Detection Item                       | Source                                                       |
| ---------- | ------------------------------------ | ------------------------------------------------------------ |
| LAKERA     | Lakera Guard                         | https://github.com/lakeraai/pint-benchmark?tab=readme-ov-file |
| Protectai  | deberta-v3-base-prompt-injection-v2  | https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2 |
| Microsoft  | Azure AI Prompt Shield for Documents | https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection#prompt-shields-for-documents |
| Meta       | Meta Prompt Guard                    | https://github.com/meta-llama/PurpleLlama/tree/main/Prompt-Guard |
| WhyLabs    | WhyLabs LangKit                      | https://github.com/whylabs/langkit                           |
| Epivolis   | Hyperion                             | https://huggingface.co/epivolis/hyperion                     |
| Fmops      | distilbert-prompt-injection          | https://huggingface.co/fmops/distilbert-prompt-injection     |
| Deepset    | deberta-v3-base-injection            | https://huggingface.co/deepset/deberta-v3-base-injection     |
| Uptrain-ai | uptrain                              | https://github.com/uptrain-ai/uptrain                        |
| Protectai  | rebuff                               | https://github.com/protectai/rebuff                          |
| Deadbits   | vigil-llm                            | https://github.com/deadbits/vigil-llm                        |




## Roadmap 📃
- [x] Prompt Injection Detection
- [x] Vector similarity detection method
- [x] Latest testing project tracking
- [ ] Model-based detection method
- [ ] Benchmarking different methods
- [ ] User Defined Detection Strategies
- [ ] Heuristics for adversarial suffixes


## License 💻

This repo is published under Apache 2.0 license and we are committed to adding more functionalities to the InjectGuard open-source repo.



## Provide feedback 📨

We are building InjectGuard in public. Help us improve by giving your feedback `gudu12122@gmail.com`