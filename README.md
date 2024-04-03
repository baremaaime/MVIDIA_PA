# Image-level micro gesture classification with 32 distinct classes.
We use a spatial cnn with ResNet101 pretrained on UCF101 dataset.
## Reference Paper
*  [Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)

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
 * Pretrained model weights can be found [here](https://drive.google.com/file/d/1Rosun5cz_3qXFyeYsBVo7lHuGzVJad_z/view?usp=drive_link).
 ```
 ./pretrained/best_model_weights.pth
 ```
 * After setting your test dataset in the appropriate directory, you can run the following script file as follows:
 ```
 sbatch run_eval.sh
 ```
 * In order to evaulate the whole testing process, you can check pyhton file `eval.py`.

 
