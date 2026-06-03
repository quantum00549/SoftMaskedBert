SoftMasked BERT Text Correction Model Implementation

Paper: Spelling Error Correction with Soft-Masked BERT

Based on PyTorch 1.9.0 and Python 3.8. This implementation refers to many GitHub implementations, fixes bugs and structural omissions from various versions, and adds extensive comments.
The dataset is Narts, or possibly SIGHAN. It was copied directly from another developer’s reproduction. If you want to use your own data, just process it according to the same format.
Some parameters can be modified in model/config.py.
Run model/train.py to train the model. The testing function is also included there. It runs without errors on CentOS 7; other platforms have not been tested yet.
The main.py file in the root directory is currently empty. I plan to write practical application methods there later, but that will be another project.



