# REDI Data Circle Project 1: 
## House Price Prediction with Multi Input Model & Mixed Data


 <p align="center">
  <img src="https://github.com/nonafalaki/Multimodal-Framework/blob/main/Structure.png" width="25%" height="25%">
 </p>
  <p align="center">
  <img src="https://pyimagesearch.com/wp-content/uploads/2019/01/keras_regression_dataset.png" width="25%" height="25%">
  </p>
  
## Sources 

* A guide how to proceed in the project can be found [here](https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/)
* The dataset can be found [here](https://github.com/emanhamed/Houses-dataset)
* Please ask for permission to use our colab [here](https://colab.research.google.com/drive/1KwWl30oYD29WTZz6e-gtM2Oj0qOB7jL9?usp=sharing#scrollTo=hEYSYBHjz5sO)

## Week 01| Preprocessing of Data

* The project analyzes houseprice prediction with mixed data. Mixed data in this case is tabular data and image data with tabular data consisting of numerical and categorical data.
* Outliers in the tabular data were removed. Also, because four photos per house exist, a single montage image from the four is arranged for all houses.

* The multimodal houseprice prediction was defined to be a regression problem.

* Training/testing splits constructed, prices scaled, and house attributes processed

 <p align="center">
  <img src="https://github.com/morschulik/multimodal_houseprice/blob/main/image1.png" width="25%" height="25%">
  </p>

## Week 02| Training of MLP with tabular data

* MLP relies on the Keras Sequential API. Our MLP is quite simple having:

   * A fully connected (Dense ) input layer with ReLU activation.
   * A fully-connected hidden layer, also with ReLU activation.
   * A regression output with linear activation. In the mixed data network we set regress=False, as regression will be performed on the entire multi-input.

* Result: 0.6 R2 score

## Week 03| Designing Multi-input + Mixed Data Model
* The create_cnn function handles the image data by accepting its parameters, filters and a boolean to append a fully-connected linear activation layer to the CNN for regression purposes.

* Thereafter we define the input via InputShape and create a set of CONV => RELU > BN => POOL layers. Each iteration of the loop appends these layers

* We Flatten the next layer and then add a fully-connected layer with BatchNormalization (BN) and Dropout (used to limit overfitting)

* Another fully-connected layer is applied to match the four nodes coming out of the multi-layer perceptron and regression node is appended

* We’ll then concatenate the mlp.output and cnn.output
 <p align="center">
  <img src="https://github.com/morschulik/multimodal_houseprice/blob/main/model.png" width="15%" height="15%">
  </p>

## Week 04| Training our Model and Hyperparameter Tuning
* Our model is compiled with "mean_absolute_percentage_error" loss and an Adam optimizer with learning rate decay

* The metrics price mean, price standard deviation, and mean + standard deviation of the absolute percentage difference are printed 

* Result: 0.66 R2 Score



