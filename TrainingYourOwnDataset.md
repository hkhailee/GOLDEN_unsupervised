# Training your own dataset steps: 

##### Inside of scan issues a bunch of users are running into the issue of not being able to put together their own dataset to train with this code base. This document hopefully helps users along their semi-supervised training on custom datasets.

## Inital work : 

* It is an issue getting the proper environment set up with this project. For that reason I am attaching my anaconda environment. (cond_env.yml) this envirnoment took a lot of time to clear of issues and is currently able to run SCAN on stl10

* git repository created with clean working code seperate branch custom_ds created to work on modifications

* execution is done through a remote cluster using slurm

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

## Reading over the paper (https://arxiv.org/pdf/2005.12320.pdf) : 

### {PRETEXT HIGHLIGHTS}
* In representation learning, a pretext task τ learns in a self-supervised fashion an embedding function Φ_θ - parameterized by a neural network with weights θ - that maps images into feature representations

* certain pretext tasks are based on specific image transformations, causing the learned feature representations to be covariant to the employedtransformation

* pretext task τ to also minimize the distance between images X_i
and their augmentations T[X_i],

*  the pretext task output is conditioned on the image, forcing Φθ to extract specific information from
its input

* Φ_θ has a limited capacity, it has to discard information
from its input that is not predictive of the high-level pretext task

### {MINING HIGHLIGHTS}
* (instance discrimination) Through representation learning, we train a model Φ_θ on the unlabeled dataset D to solve a pretext task τ

* for every sample X_i ∈ D, we mine its K nearest neighbors in the embedding space Φ_θ

* set N_X_i as the neighboring samples of X_i in the dataset D

* mined nearest neighbors are instances of the same semantic cluster to a degree

* see loss function page 5
    - the dot product will be maximal when the predictions are one-hot (confident) and assigned to the same cluster (consistent)
    -  the entropy spreads the predictions uniformly across the clusters C

* approximate the dataset statistics by sampling batches of sufficiently large size

*  randomly augment the samples X_i and their neighbors N_X_i
<br />
note not all mined neighbors are classified into the same cluster.samples with highly confident predictions (pmax ≈ 1) tend to be classified to the proper cluster.

### {SELF LABELING HIGHLIGHTS}

* during training confident samples are selected by thresholding
the probability at the output, i.e. p_max > threshold
* For every confident sample, a pseudo label is obtained by assigning the sample to its predicted cluster
* A cross-entropy loss is used to update the weights for the obtained pseudo labels
* To avoid overfitting, we calculate the cross-entropy loss on strongly augmented
versions of the confident samples
*   The self-labeling step allows the network to
correct itself, as it gradually becomes more certain, adding more samples to the
mix.

### Experiments 
* The results are reported as the mean and standard deviation from 10 different runs. Finally, all experiments are performed using the same backbone, augmentations, pretext task and hyperparameters.
    -  standard ResNet-18 backbone
    - SimCLR implementation for the instance discrimination task on the smaller datasets
    -  MoCo on ImageNet (large)
##### training
* every image is disentangled as a unique instance independent of the applied transformation
* transfer the weights, obtained from the pretext task to initiate the
clustering step
* clustering step for 100 epochs using batches of size 128
*  weight on the entropy term is set to λ = 5. A higher weight avoids the premature grouping of samples early on during training
* After the clustering step, we train for another 200 epochs using the self-labeling procedure with threshold 0.99
* weighted cross-entropy loss compensates for the imbalance between confident samples across clusters
* The network weights are updated through Adam with learning rate 10^−4 and weight decay 10^−4
* during both the clustering and selflabeling steps images are strongly augmented by composing four randomly selected transformations from RandAugment 
<br />
keeps mentioning supplimentary materials wish they would reference specifically which ones. (SM starts on page 18)

##### validation criteria
* During the clustering step, select the best model based on the lowest loss.
* During the self-labeling step, save the weights of the model when the amount of confident samples plateaus (reach a state of little or no change after a time of activity or progress.)
* __We follow these practices as we do not have access to a labeled validation set.__

* applying K-means to the pretext features outperforms prior state-of-the-art methods for unsupervised classification based on end-to-end learning schemes
* Updating the network weights through the SCAN-loss - while augmenting the input images through SimCLR transformations - outperforms K-means
    -  SCAN-loss avoids the cluster degeneracy issue
* Applying transformations from RandAgument (RA) to both the samples and their mined neighbors further improves the performance
* Fine-tuning the network through self-labeling further enhances the quality of the cluster assignments
    - in order for self-labeling to be successfully applied, a shift in augmentations is required i.e. 
        - applying transformations from RandAgument

* results not very sensitve to the number of mined neighbors, stays nearly consistent from 5-50 preformance only decreases as we approach 0.
* The lower performance improvement on CIFAR100-20 can be explained by the ambiguity of the superclasses used to measure the accuracy
    - there is not exactly one way to group categories like omnivores or carnivores together

##### evaluation 
*  evaluate the results based on clustering accuracy (ACC), normalized mutual information (NMI) and adjusted rand index (ARI)
*  did not have to perform any dataset specific fine-tuning
* a bit quicker than other methods  

### overclustering
* assumed to have knowledge about the number of ground-truth classes.
* method predictions were evaluated using a hungarian matching algorithm
*  conclude that the approach does not require knowledge of the exact number of clusters

##### imageNET

* Training the model with the SCAN-loss again outperforms the application of K-means
* results are further improved when fine-tuning the model through self-labeling
* Table 5 compares our method against recent semi-supervised learning approaches when using 1% of the images as labelled data (similar to our enrico?)
* method outperforms several semi-supervised learning approaches, without using labels. This further demonstrates the strength of our approach

hopping back in
----------------------------------------------------------------------

# Execution PRETEXT:

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

(paper page 19)
__Pretext task__ select instance discrimination as our pretext task. In particular, we use the implementation from MoCo [8]. We use a ResNet-50 model as backbone

__clustering step__  freeze the backbone weights during the clustering step, only train the final linear layer using the SCAN-loss, train ten separate linear heads in parallel, when initiating the self-labeling step, we select the head with the lowest loss to continue training,Every image is augmented using augmentations from SimCLR, We reuse the entropy weight from before (5.0), and train with batches of size 512, 1024 and 1024 on the subsets of 50, 100 and 200 classes respectively, y. We use an SGD optimizer with momentum 0.9 and initial learning rate 5.0. trained for 100 epochs. On the full ImageNet dataset, we increase the batch size and learning rate to 4096 and 30.0 respectively, and decrease the number of neighbors to 20

__self labeling__ strong augmentations from RandAugment to
finetune the weights through self-labeling, __followed by Cutout__. The model weights are updated for 25 epochs using SGD with momentum 0.9. The initial learning rate is set to 0.03and kept constant. Batches of size 512 are used. Importantly, the model weightsare updated through an exponential moving average with α = 0.999. We did not find it necessary to apply class balancing in the cross-entropy loss



### thoughts 01: 
Initially going through these I believed that stl-10 would be the best option since it does include an unlabeled binary and nearly all of the data is unlabeled. However, I think imageNet needs a closer look. With SCAN there is already completed pretext processing, scan and selflabel training that has happened on the imageNet data. This dataset spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images of everyday nouns. 

<br />
even if our labels do not currently exist in this trained model I believe we can customize it to include only our labels. This will likely be more informed than just using our own pretrained model. 

__should be tested down the line for comparison__

* I have decided to mimick imageNet
## Evaluating imageNet for Mimicking (Data and Model)

Right off the bat in the __getitem__ method I am not seeing any issues. 
<br />

__Note : image size is 256 x 256__

* it has a split = 'train' 
* the files are in .jpeg format
* line 57  Read the subset of classes to include (sorted) called subset_file

* the yml file for all the moco_imageNetXXXX.yml have the same normailzation points

Simple enough so far. I am now going to evaluate my own dataset.

## Evaluation of Custom Dataset RICO

* enRicos images are all different sizes, these are being sampled from the RICO which implies RICO also does not have uniform sizes. This is fine as the code later will be resizeing these images. evaluation done with basic PIL image size command. 

* checking the number of channels and ensuring that conversions inside of imageNet will be accurate on the RICO data (this is done inside evaluate_rico.py)

After experimenting it was found that all of the images in the RICO dataset are of different sizes and of different color scales, the methodology presented inside of data/imagenet.py effectivly transforms the given size of an image. however if the image is in gray scale it will not convert it to true rgb, however it will continue to have the apropriate 3 channels. Output stats of evaluate_rico.py : 

- starting number of gray images 20
- starting number of color images 2644
- final number of gray images (should be 0) 20
- final number of color images (should be all) 2644
- total reformed images 256by256 1342
- total non reformed images (should be 0) 0

Even though the images are still in gray scale this is fine because there is still a working 3 channels.

moving on to mimicking imageNet...

## Steps for Utils/common_config.py

* __get_model__ elif p['backbone'] == 'resnet50':
    - `elif 'rico-20' in p['train_db_name']:`
    - `from models.resnet import resnet50`
    - `backbone = resnet50()`


* __get_train_dataset__ 
    - `elif p['train_db_name'] =='rico-20':`
    - `from data.rico import RICO20`
    - `'./data/rico_subsets/%s.txt'`
    - `dataset = RICO20(subset_file=subset_file, split='train', transform=transform)`

* __get_val_dataset__
    - `elif p['val_db_name'] == 'rico-20':`
    - `from data.rico import RICO20`
    - `subset_file = './data/rico_subsets/%s.txt' %(p['val_db_name'])`
    - `dataset = RICO20(subset_file=subset_file, split='val', transform=transform)`

### thoughts 02 -- imageNet subsets: 
I mentioned before in thoughts 01 that I was going to try to use the already established imageNet trained model. At this point in time I am not seeing a way to do that, so I will start implementing a similar one.

note : p['augmentation_strategy'] == 'ours': exists. also format of imagenet_XXX.txt: <br />
n01558993 robin, American robin, Turdus migratorius <br />
n01601694 water ouzel, dipper <br />
n01669191 box turtle, box tortoise <br />

## Steps for data/rico.py

* two folders val and train 
    - `self.root = os.path.join(root, '%s/' %(split))` 
* added class information to ./data/rico_subsets/rico-20.txt
* for gathering the files sorted I removed the subdir loop, only called the train directory itself to get all of the files. (lines 29-34 in data/rico.py)

* per suggestion of SCAN I made a testing set of 19500 images
* ... need to seperate the val set into different folders. while the training one might not use it and thats fine, val does and they use the same data file. 
    - created 20 different class directories with images in them the same verbal name 

## Revaluation: 
After reading through the whole paper again, it has been reinforced that all the methodologies should be done to find best results. 

## Steps for data/rico.py CONT. 

modifying the previous implementation from RICO20 to RICO20_sub. made class RICO20 with no subsets training on train folder in rico_image

* included a subset_file argument to not have to change utils/common_config.py again

## Steps configs/pretext/moco_rico.py

Straight copy over- changed db names to rico-20. 
* batch size modified to 128 for space limitations
* changed num classes to 20

I think thats it for pretext...

#### HIGHKEY NOTE: since mimicking imagenet to run pretext we will use moco and not simclr
Additionally. Say you have your images in ./images/train/ for moco.py to see them all the images need to be placed in a subdirectory inside of train. for example ./images/train/1/ with the mypath.py still only pointing to ./images/ and the dataload file still only specifying ./images/train/

## Pretext Results 

Accuracy of top-50 nearest neighbors on train set is 100.00
[34mMine the nearest neighbors (Val)(Top-5)[0m
Fill Memory Bank [0/11]
Mine the neighbors
Accuracy of top-5 nearest neighbors on val set is 32.91

# Implementing Semantic CLusteriing task on custom dataset

Starting off I think the first thing to do is create a configs/scan for rico-20

## Steps for configs/scan/scan_rico-20.yml 

* straight copied over scan_imagenet_50.yml 
    - modified to rico-20 dataset
    - changed num_classes to 20 
    - kept num neighbors at 50 (since it doesnt really affect)
    - changed batch size to 128 and epochs to 300 (proportionate)

No modification needed to scan.py. During scan traing number of epochs remained at 100 even with the modified environment file.

sucessfully ran scan.py

## Steps for configs/selflabel/selflabel_rico-20.yml 

* straight copied over selflabel_imagenet_50.yml
    - changed epchs to 75
    - batch size 128
    - dataset to rico-20





