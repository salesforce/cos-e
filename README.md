# Commonsense Explanations (CoS-E)

This repository contains human commonsense explanations for the [Commonsense Question Answering (CQA)](https://www.tau-nlp.org/commonsenseqa) dataset. We collected explanations for the train and dev sets for v1.0 and v1.11 of CQA. The instance ids correspond to questions in CQA.

Each file has two types of explanations:
1. Selected: the highlighted text span in the question that serves as justification for the answer choice.
2. Open-ended: free-form natural language explanation.

Along with the raw files, we also included the post-processed files for each of the categories. However, note that we used the original raw files for all our experiments in our paper.

#### Ethical concerns
We observed substantial gender disparity and bias in the CQA dataset with higher proportion of female pronouns used in negative contexts. This kind of bias has inevitably propagated into CoS-E. We advise that these datasets and trained models be
used with that in mind.

On a positive note, because CoS-E uses crowd-sourcing it also adds diversity of perspective and in particular diverse reasoning on world knowledge to the CQA dataset.

### Bibtex
If you use this dataset or paper in your work, please cite us:

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
