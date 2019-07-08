# Docuwarp
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7acd5aa8048a4c96bd97a96bac2639d1)](https://app.codacy.com/app/huang836/deep-learning-for-document-dewarping?utm_source=github.com&utm_medium=referral&utm_content=thomasjhuang/deep-learning-for-document-dewarping&utm_campaign=Badge_Grade_Dashboard)
This project is focused on dewarping document images through the usage of pix2pixHD. The objective is to take images of documents that are warped, folded, crumpled, etc. and convert the image to  use the official pix2pixHD repository to train and perform inference. 

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 

### Code

Template code is provided in the `boston_housing.ipynb` notebook file. You will also be required to use the included `visuals.py` Python file and the `housing.csv` dataset file to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project. Note that the code included in `visuals.py` is meant to be used out-of-the-box and not intended for students to manipulate. If you are interested in how the visualizations are created in the notebook, please feel free to explore this Python file.

### Training

In a terminal or command window, navigate to the top-level project directory (that contains this README) and run one of the following commands:

```bash
python train.py --name proj_name --dataroot ./datasets/proj_name/ --label_nc 0 --resize_or_crop crop --no_instance --no_flip
```  

Use flag --fp16 if you have NVIDIA Apex installed, and wish to use Automatic Mixed Precision, this improves training speed by up to 80% at best.

### Data

The modified Boston housing dataset consists of 489 data points, with each datapoint having 3 features. This dataset is a modified version of the Boston Housing dataset found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing).

**Features**
1.  `RM`: average number of rooms per dwelling
2. `LSTAT`: percentage of population considered lower status
3. `PTRATIO`: pupil-teacher ratio by town

**Target Variable**
4. `MEDV`: median value of owner-occupied homes
