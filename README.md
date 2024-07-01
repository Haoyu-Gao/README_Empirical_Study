# README_Empirical_Study

This is the GitHub repository replication package for paper **"Add more config detail": A Taxonomy of Installation Instruction Changes** under submission to IEEE Transactions on Software Engineering.  


## Folders walkthrough
### Code folder
Folder `Code` contains several components, including mining software repositories (`Code/data_collection`), and README commit filter and statistical analysis (`Code/analysis`). Folder `Code/qualitative_analysis_samples` contains our cluster sample repositories splitted into four buckets. Folder `Code/commit_annotation` gives the annotation of all commits for our samples as "relevant" or "irrelevant" 

Furthermore, keywords and headers that we annotated as "relevant" and "irrelevant" is available under this [folder](Code/analysis/annotated_headers)

### Qualitative Result Examples
In our results analysis, we provide 35 examples explaining our codes and categories. We provide the screenshots with corresponding example IDs. These examples and the raw URLs to the original GitHub page are provided in file `Qualiatative Result Example/result_examples.xlsx`. 

If you are interested in our coding process, or would like to reuse our data, `Qualiatative Result Example/Qualitative Analysis Raw Data and Codes.xlsx` provides two spreadsheets. Sheet `RQ-instance-coding` gives information on how we assign codes to each piece of data. Sheet `RQ-code-concept-category` provides the detail of each code, concept and category.``

### Survey Validation
In the paper, we provided a README template, aiming to help practitioners perform documentation maintenance. To validate this template, we applied this template on raw documents to form "augmented documents". These documents were provided to online participants with IT backgrounds to evaluate on 10 dimensions. The documents as well as a screenshot of one survey is attached in this folder. The participants' response is recorded and organised in `sample.xlsx`.Please refer to our paper for more details.

### Citation
If you use our provided data or results, please cite our paper
```
@article{gao2023add,
  title={" Add more config detail": A Taxonomy of Installation Instruction Changes},
  author={Gao, Haoyu and Treude, Christoph and Zahedi, Mansooreh},
  journal={arXiv preprint arXiv:2312.03250},
  year={2023}
}
```
