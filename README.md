# mini_lab_to_experiment_on_learning_schedule

you should create a folder under the name {DATA_0}
and under it :
  empty : store empty images {white images} // train on them
  empty_evaluate : store empty images {white images} // test on them
  notempty : store not empty images {at least one black pixel} // train on them
  notempty_evaluate : store not empty images {at least one black pixel} // test on them
then you should run the data_0 generator code for creating the number of images you want in training by keeping ev = "" (empty string) and update the number of samples corespending to it 
but for the evaluation images you should create 1000 images by setting ev = "_evaluate"
