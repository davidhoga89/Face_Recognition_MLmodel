Prodigious Latam Face Recognition Science Fair Project Readme

Steps:
1 - face_detection.py 
is the script to test the camera and the face detection code with its respective logging (which needs to be implemented into the final recogniser file)

2 - data_gathering.py 
Run in order to acquire a new face/id to be recognised by the model.
For future applications should be run every time an unknown face is detected as a parallel service.
*IMPORTANT! In Mac or windows there are hidden files (.DS_data, etc) in the dataset folder that the code reads and throws an error since it is not an image. This is generated every time a new file is added. Therefore we need to modify this code to only scan the files containing the *.jpg/png extension

3 - trainer.py
This script should be run every time new users (face images) are added into the dataset.
This generates a file called trainer.yml, which is stored in the trainer folder.
Again this can be added as a background service or run periodically (Every hour for instance) or whenever more info is added to the dataset (for example comparing the total quantity of images in the folder, or size of array, and whenever the value is different, run it). 

4 - recognizer.py
Once the model is trained (trainer.yml generated) this is the final file that recognizes faces.
We need to try the scenario when the model is retrained, the windows doesn’t get affected, broken or damaged. (So when the trainer.yml is overwritten) maybe thinking of closing and reopening again.
Still have some work to do, such as the logging integration, etc.

VOILA!!!

Next step could be re run the data_gathering.py script, or create another one, whenever it detects an identified user to feed back the dataset with more faces (5 images could be, not 30 since wouldn’t have the time to take that amount of snapshots) and keep training the model to make it more efficient and confident identifying people.