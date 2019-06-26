# Commonsense Explanations (CoS-E)

This repository contains human commonsense explanations for the [Commonsense Question Answering (CQA)](https://www.tau-nlp.org/commonsenseqa) dataset. We collected explanations for the train and dev sets for v1.0 and v1.11 of CQA. The instance ids correspond to ids in the CQA datasets.

Each file has two types of explanations:
1. Selected: the highlighted text span in the question that serves as justification for the answer choice.
2. Open-ended: free-form natural language explanation.

Along with the raw files, we also included the post-processed files for each of the categories. However, note that we used the original raw files (even though noisy) for all our experiments in our paper.


#### Ethical concerns
We observed substantial gender disparity and bias in the CQA dataset with higher proportion of female pronouns used in negative contexts.

Following are some such examples from CQA:
- **Q:** "She was a horrible pet owner, she put a what on her cat?" <br>
   **AC:** **wool sweater,** get wet, eat vegetables
- **Q:** "The woman was yelling obscenities in public,and while it was entertaining for people passing by,what did her husband feel?"<br>
  **AC:** fulfillment, fatigue, **embarrassment**

This kind of bias has inevitably propagated into CoS-E. We advise that these datasets and trained models be
used with that in mind.

On a positive note, because CoS-E uses crowd-sourcing it also adds diversity of perspective and in particular diverse reasoning on world knowledge to the CQA dataset.

### Bibtex
If you use this dataset or paper in your work, please cite [Explain Yourself! Leveraging Language models for Commonsense Reasoning](https://arxiv.org/abs/1906.02361):

```
@inproceedings{rajani2019explain,
     title = "Explain Yourself! Leveraging Language models for Commonsense Reasoning",
    author = "Rajani, Nazneen Fatema  and
      McCann, Bryan  and
      Xiong, Caiming  and
      Socher, Richard",
      year="2019",
    booktitle = "Proceedings of the 2019 Conference of the Association for Computational Linguistics (ACL2019)",
    url ="https://arxiv.org/abs/1906.02361"
}
```
