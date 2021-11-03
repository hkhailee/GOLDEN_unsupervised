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

---------------------------------------------------------------------