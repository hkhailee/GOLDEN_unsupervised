# Training your own dataset steps: 

##### Inside of scan issues a bunch of users are running into the issue of not being able to put together their own dataset to train with this code base. This document hopefully helps users along their semi-supervised training on custom datasets.

## Inital work : 

* It is an issue getting the proper environment set up with this project. For that reason I am attaching my anaconda environment. (cond_env.yml) this envirnoment took a lot of time to clear of issues and is currently able to run SCAN on stl10

* git repository created with clean working code seperate branch custom_ds created to work on modifications

* execution is done through a remote cluster using slurm

# Real work:

According to the previous questions and answers in the SCAN (https://github.com/wvangansbeke/Unsupervised-Classification) repository issues

##### steps:
1. add location of dataset to utils/mypath.py

2. create a class for your dataset similar to the ones in the data folder 
    * specifically pay attention to __getitem__

3. create a config file for 
    * configs/pretext 
    * configs/scan 
    * configs/selflabel 

4. If you do not want to use labels for evaluation you need to remove the parts from the dataloader
    * You do not need base_dataset or base_dataloader this is because you will have no val_transforms.
    * These are only needed for valiadtion to evaluate with kNN the complete training set is used (i.e. memory_bank_base)
    * If you remove the validation part you cannot compute accuracy 
    * Validation loss is used to select the best model, you can define your own validation set or simply take the final model
    * Lowering weight in the loss will likely help with consistency loss close to entropy and probabilities being very close together.

# Execution:

To be transparent I am using the RICO (https://interactionmining.org/rico) and enRICO (https://github.com/luileito/enrico) datasets. RICO contains around 66k unlabled images of user interfaces. enRICO is a subset of RICO containting around 1.2k labeled images of the UI's. 

I want to be able to use SCAN on the unlabled data and with semi supervised training use validation through the pretext processing stage on the labeled enRICO data.

|Topic name     |Num. UIs       |Description                        |
|---------------|---------------|-----------------------------------|
|Bare	        |76	            |Largely unused area                |
|Dialer         |6	            |Number entry                       |
|Camera	        |8	            |Camera functionality               |
|Chat	        |11	            |Chat functionality                 |
|Editor	        |18	            |Text/Image editing                 | 
|Form	        |103        	|Form filling functionality         |
|Gallery	    |144	        |Grid-like layout with images       |
|List	        |265	        |Elements organized in a column     |
|Login	        |141	        |Input fields for logging           |
|Maps	        |9	            |Geographic display                 |
|MediaPlayer	|32	            |Music or video player              |
|Menu	        |79	            |Items list in an overlay or aside  |
|Modal	        |67	            |A popup-like window                |
|News	        |59	            |Snippets list: image, title, text  |
|Other	        |52	            |Everything else (rejection class)  |
|Profile	    |63	            |Info on a user profile or product  |
|Search	        |35	            |Search engine functionality        |
|Settings	    |90	            |Controls to change app settings    |
|Terms	        |39	            |Terms and conditions of service    |
|Tutorial	    |163	        |Onboarding screen                  |

------------------------------------------------------

* I have set the `root_dir` in `env.yml` to 
    - `/path/to/scratch/imageClassification/GOLDEN/Unsupervised-Clasifciation/RESULTS_RICO20/` 
    - This is where all of the training blocks are going to be stored.

 * The next step would be to add my dataset path to utils/mypath.py 
    - To keep things simple I am going to mimick the current in place datasets but ultimatly delete them in the end on this branch to keep things simple.
    - code added `if database == 'rico-20': return '/path/to/rico-20/'`

* The issue now is evaluating how the images are going to be dataloaded. I am going to figure this out first before actually moving all my images over to this directory. 
    - `data/rico.py` created
    - I believe the next step will be to evaluate the other datasets and see which is the most similar to what I am trying to go for. 
    <br />
##### cifar: (https://www.cs.toronto.edu/~kriz/cifar.html)

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

`def unpickle(file):`
    `import pickle`
    `with open(file 'rb') as fo:`
        `dict = pickle.load(fo, encoding='bytes')`
    `return dict`
* __data__ -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

* __labels__ -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

* __label_names__ -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, `label_names[0] == "airplane", label_names[1] == "automobile",...` etc.

##### stl: (https://cs.stanford.edu/~acoates/stl10/)

The STL-10 dataset is an image recognition dataset for developing unsupervised feature learning, deep learning, self-taught learning algorithms. It is inspired by the CIFAR-10 dataset but with some modifications. In particular, each class has fewer labeled training examples than in CIFAR-10, but a very large set of unlabeled examples is provided to learn image models prior to supervised training. The primary challenge is to make use of the unlabeled data (which comes from a similar but different distribution from the labeled data) to build a useful prior. __We also expect that the higher resolution of this dataset (96x96) will make it a challenging benchmark for developing more scalable unsupervised learning methods.__

* 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
* Images are 96x96 pixels, color.
* 500 training images (10 pre-defined folds), 800 test images per class.
* 100000 unlabeled images for unsupervised learning. 
* These examples are extracted from a similar but broader distribution of images. For instance, it contains other types of animals (bears, rabbits, etc.) and vehicles (trains, buses, etc.) in addition to the ones in the labeled set.
* Images were acquired from labeled examples on ImageNet.

__testing protocol__
<br />
We recommend the following standardized testing protocol for reporting results:

1. Perform unsupervised training on the unlabeled.
2. Perform supervised training on the labeled data using 10 (pre-defined) folds of 100 examples from the training data. The indices of the examples to be used for each fold are provided.
3. Report average accuracy on the full test set.

what is include:

* url of the binary data
    - DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

images are stored in column-major order i.e. the first 96*96 values are the red channel, the next 96*96 are green, and the last are blue

<br />

__labels are in the range 1 to 10__

__data files__ : train_X.bin, test_X.bin, unlabeled.bin
__label files__ : train_y.bin, test_y.bin
* __class_names.txt__ file is included for reference, with one class name per line.
* __fold_indices.txt__ contains the (zero-based) indices of the examples to be used for each training fold. The first line contains the indices for the first fold, the second line, the second fold, and so on.

* reading the images with python (https://github.com/mttk/STL10/blob/master/stl10_input.py)

##### imagenet (https://image-net.org/)

ImageNet is an image dataset organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet; the majority of them are nouns (80,000+). In ImageNet, we aim to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated. In its completion, we hope ImageNet will offer tens of millions of cleanly labeled and sorted images for most of the concepts in the WordNet hierarch. 


### thoughts : 
Initially going through these I believed that stl-10 would be the best option since it does include an unlabeled binary and nearly all of the data is unlabeled. However, I think imageNet needs a closer look. With SCAN there is already completed pretext processing, scan and selflabel training that has happened on the imageNet data. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images of everyday nouns. 

<br />
even if our labels do not currently exist in this trained model I believe we can customize it to include only our labels. This will likely be more informed than just using our own pretrained model. 

__should be tested down the line for comparison__

* I have decided to mimick imageNet
# Evaluating imageNet for Mimicking (Data and Model)

Right off the bat in the __getitem__ method I am not seeing any issues. 
<br />

__Note : image size is 256 x 256__

* it has a split = 'train' 
* the files are in .jpeg format
* line 57  Read the subset of classes to include (sorted) called subset_file

* the yml file for all the moco_imageNetXXXX.yml have the same normailzation points

Simple enough so far. I am now going to evaluate my own dataset.

# Evaluation of Custom Dataset RICO

* enRicos images are all different sizes, these are being sampled from the RICO which implies RICO also does not have uniform sizes. This is fine as the code later will be resizeing these images. evaluation done with basic PIL image size command. 

* checking the number of threads and ensuring that conversions inside of imageNet will be accurate on the RICO data (this is done inside evaluate_rico.py)

After experimenting it was found that all of the images in the RICO dataset are of different sizes and of different color scales, the methodology presented inside of data/imagenet.py effectivly transforms the given size of an image. however if the image is in gray scale it will not convert it to rgb. Output stats of evaluate_rico.py : 

- starting number of gray images 20
- starting number of color images 2644
- final number of gray images (should be 0) 20
- final number of color images (should be all) 2644
- total reformed images 256by256 1342
- total non reformed images (should be 0) 0

Even though the images are still in gray scale this is fine because there is still a working 3 channels.

moving on to mimicking imageNet...

