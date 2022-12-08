# Shot-Predictor
## ECE379K Final Project

In basketball, the ability of a player to effectively shoot the basketball typically comes down to the player’s shooting form. While the form of the best shooters tend to look different, they all typically use the same fundamentals. In our project, we will attempt to capture these fundamental aspects of a player’s shooting form and attempt to predict the outcome of a shot using these features. Research has been done on extracting features from a player’s movement to classify the action a player is performing (shooting, dribbling, etc.) [1], but we would like to focus our energy on feature extraction from the shooting motion using pose estimation [2], object detection [3], and possibly other methods to extract feature descriptors of a shot and attempt to identify it as a make or a miss.


References:

[1] R. Ji, "Research on Basketball Shooting Action Based on Image Feature Extraction and Machine Learning," in IEEE Access, vol. 8, pp. 138743-138751, 2020, doi: 10.1109/ACCESS.2020.3012456.

[2] Cao, Z., Hidalgo, G., Simon, T., Wei, S., & Sheikh, Y. (2021). OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43, 172-186.

[3] J. Redmon, S. Divvala, R. Girshick and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779-788, doi: 10.1109/CVPR.2016.91.

---
## How to Run Demo Code
First, clone the repository with
```bash
git clone https://github.com/afnanmmir/Shot-Predictor.git
```
or
```bash
git clone git@github.com:afnanmmir/Shot-Predictor.git
```
Then, you should create a virtual environment to install required packages in. After creating and activating the virtual environment, run
```bash
pip install -r requirements.txt
```
to install the required packages. You can then run the demo code that is in the `models/demo/` directory. Make sure you run the code with the python kernel of the python environment that you just created.
