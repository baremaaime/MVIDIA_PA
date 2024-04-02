# Image-level micro gesture classification with 32 distinct classes.
We use a spatial cnn with ResNet101 pretrained on UCF101 dataset.
## Reference Paper
*  [[1] Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)

## 1. Data
  ### 1.1 Spatial input data -> rgb frames
  * In order to run test on new data, make sure you organize the test data as follows:
  ```
  ./root/test
  |
  |───1/
  │    └───xxx.jpg
  │    └───...
  └───2/
  │    └───xxx.jpg
  │    └───...
  └───3/
  │    └───xxx.jpg
  │    └───...
  ```

## 2. Testing 
  ### Spatial stream
 * After setting your test dataset in the appropriate directory, you can run the following script file as follows:
 * Training and testing
 ```
 sbatch run_eval.sh
 ```
 * In order to evaulate the whole testing process, you can check pyhton file `eval.py`.

 
