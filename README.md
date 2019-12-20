# Commonsense Explanations (CoS-E) Dataset

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

## Code to reproduce results
We have released all the code to reproduce results from our ACL paper for both v1.0 and v1.11 of the dataset. There are two main files, one for generating explanations using GPT and another is the BERT classifier for CQA. Both these files depend on the [Huggingface transformers repository] (https://github.com/huggingface/transformers) and because there have been many changes to the huggingface repo, we recommend the following steps:

1. Create a new conda env using `conda create -n cose python=3.7`.
2. Activate conda env using `conda activate cose`.
3. Checkout the huggingface transformers commit that works with our code assuming you already have cloned the repo.
`git checkout e14c6b52e37876ee642ffde49367c51b0d374f41`.
4. Copy all the code to the `examples` directory and make sure your `PYTHONPATH` is set correctly to point to `pytorch_pretrained_bert` for all the imports to work. 

### Getting data ready
`parse-commonsenseQA.py` allows you to convert the jsonl files to the csv format that both the explanation generation and CQA classifier require. 

**Example Usage:** `python parse-commonsenseQA.py <CQA-train-file> <cose-train-file> <output-csv-file>`

### Generating commonsense explanations using GPT
Takes the csv files as input from the previous steps and fine-tunes a conditional LM to generate explanations. These are the minimal arguments that the code requires but you can also set other parameters listed in the file.

**Example Usage:** `PYTHONPATH=../:$PYTHONPATH python train_commonsenseqa_v1.1.py --do_train --output_dir out --data ../data/`

### Classifying CQA with explanations using BERT
Once you have the LM explanations, you can paste it as a column to the csv file and make sure the code points to the correct explanation column (human vs. LM) that you are interested in using.

**Example Usage:** `PYTHONPATH=../:$PYTHONPATH python run_commonsenseQA_expl_v1.1.py --do_train --output_dir out --data ../data/`

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
