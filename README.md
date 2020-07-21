# TREC Deep Learning Quick Start

This is a quick start guide for the document ranking task in the TREC Deep Learning (TREC-DL) benchmark.
If you are new to TREC-DL, then this repository may make it more convenient for you to download all the required datasets and then train and evaluate a relatively efficient deep neural baseline on this benchmark, under both the rerank and the fullrank settings.

If you are unfamiliar with the TREC-DL benchmark, then you may want to first go through the websites and overview paper corresponding to previous and current editions of the track.
* TREC-DL 2019: [website](https://microsoft.github.io/TREC-2019-Deep-Learning) and [overview paper](https://arxiv.org/pdf/2003.07820.pdf)
* TREC-DL 2020: [website](https://microsoft.github.io/TREC-2020-Deep-Learning/) (Currently open for submissions!)

### DISCLAIMER
While some of the contributors to this repository also serve as organizers for TREC-DL, please note that this code is **NOT** officially associated in any way with the TREC track.
Instead, this is a personal codebase that we have been using for our own experimentation and we are releasing it publicly with the hope that it may be useful for others who are just starting out on this benchmark.

As with any research code, you may find some kinks or bugs.
Please report any and all bugs and issues you discover, and we will try to get to them as soon as possible.
If you have any questions or feedback, please reach out to us via [email](mailto:bmitra@microsoft.com) or [Twitter](https://twitter.com/UnderdogGeek).

Also, please be aware that we may sometimes push new changes and model updates based on personal on-going research and experimentation.


## The Conformer-Kernel Model with Query Term Independence (QTI)

The base model implements the Conformer-Kernel architecture with QTI, as described in this [paper]().

![The Conformer-Kernel architecture with QTI](images/CK.png)

If you use this code for your research, please cite the [paper]() as follows:

```
@article{mitra2020conformer,
    title={Conformer-Kernel with Query Term Independence for Document Retrieval},
    author={Mitra, Bhaskar and Hofstatter, Sebastian and Zamani, Hamed and Craswell, Nick},
    journal={arXiv preprint arXiv:},
    year={2020}
}
```

Specifically, the code provides a choice between three existing models:
* **NDRM1**: A Conformer-Kernel architecture with QTI for latent representation learning and matching
* **NDRM2**: A simple learned BM25-like ranking function with QTI for explicit term matching
* **NDRM3** (default): A linear combination of **NDRM1** and **NDRM2**

You can also plug-in your own neural model by simply replacing the ```model.py``` and ```model_utils.py``` with appropriate implementations corresponding to your model.
The full retrieval evaluation assumes query term independence.
If that assumption does not hold for your new model, please comment out the calls to ```evaluate_full_retrieval``` in ```learner.py```.

## Requirements

The code in this repository has been tested with:
* **Python version**: 3.5.5
* **PyTorch version**: 1.3.1
* **CUDA version**: 10.0.130

The training and evaluatin were performed using **4 Tesla P100 GPUs** with 16280 MiB memory each.
Depending on your GPU availability, you may need to set the minibatch size accordingly for train (```--mb_size_train```), test (```--mb_size_test```), and inference (```--mb_size_infer```).

In addition, the code assumes the following Python packages are installed:
* numpy
* fasttext
* krovetzstemmer
* clint

Using PIP, you can install all of them by running the following from command-line:

```
pip install numpy fasttext krovetzstemmer clint
```

## Getting Started

Please clone the repo and run ```python run.py```.

The script should automatically download all necessary data files, if missing, which can take significant amount of time depending on network speed.
If the download fails for any particular file then please delete the local incomplete copy and re-run the script.
The script performs pretty aggressive text normalization that may not always be appropriate.
Please be aware of this and modify the code if you desire a different behaviour.

After the download completes, the script should first pretrain a word2vec model for the input embeddings.
Then subsequently, it should train a simple neural document ranking model (NDRM) and report metrics on the TREC-DL 2019 test set for both the reranking and the fullranking tasks.
The script should also prepare the run files corresponding to the TREC-DL 2020 test set for submission.

Couple of additional notes:
* The code automatically downloads the whole ORCAS dataset.
I plan to make this optional in the future but have not got to implementing it yet.
So, please feel free to disable that in the code directly for now to avoid downloading them unnecessarily if you don't plan to use them.
* The IDF file is generated conservatively only for terms that appear in the train, dev, validation, and test queries.
So, if you change or add to the query files, then please delete the generated IDF file and rerun the script to regenerate it.

## Legal Notices

Microsoft and any contributors grant you a license to the Microsoft documentation and other content in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode), see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the [LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at <http://go.microsoft.com/fwlink/?LinkID=254653>.

Privacy information can be found at <https://privacy.microsoft.com/en-us/>.

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.
