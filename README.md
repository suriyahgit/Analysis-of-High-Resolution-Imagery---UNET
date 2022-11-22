
# **Building segementation using Unet Algorithm**

Unet is a special type of Convolutional Neural Network and by using a U-Net Semantic image segmentation can be achieved. Which is common and effective way to deal with this building detection and segmentation problems. U-Net also allows us to go above and beyond normal image classification and object detection, to classify the pixels of those objects in their exact shape. Hence we planned to use Unet instead of other deep learning methods.

## **GIT path**

- Git path - **https://github.com/suriyahgit/Analysis-of-High-Resolution-Imagery---UNET**


## **Getting Started**

These instructions will get you a copy of the project up and running on your local machine. 

### Cloning Project

- Clone the Project to the development system

  `https://github.com/suriyahgit/Analysis-of-High-Resolution-Imagery---UNET.git`

### Prerequisites

-Install Python 3.8

### Virtual Environment Setup

#### Install virtualenv pip package

- Install Virtual Environment using pip

  `pip install virtualenv`

#### Creating virtual environment

- Create Virtual Environment 'env' as follows in the home path of the repository

  `virtualenv <env_name>`

#### Activate the Virtual Environment in Windows

- Use the following command to activate the environment

  `<path_to_environment>\Scripts\activate`

#### Install project dependancies using pip

- Activate the virtual environment in the repository home path and run the below command from project path where requirements.txt is available to install the pip packages.

  `pip install -r requirements.txt`

#### Development Guidelines

#### Procedure

- Perform all the above mentioned steps. Then download the data from the data folder from the github.
- Run the unet.py.
- If you get system out of memory error. Try to run with 'GPU'.
- To avoid this error, we have used "Google Colab" to run the script. 
- Inorder to use it in colab, download the data and zip the data folder.
- Upload in your drive and make the path changes in the script.
- In colab, >runtime >change runtime >type = "GPU" and then run it.

#### Usual Errors

- Dependency conflict errors
- DataLoader errors: This is because of the lack of support of GPU in the system the program is run
- Path contradictions
