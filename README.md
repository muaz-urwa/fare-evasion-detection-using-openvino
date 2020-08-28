
## clone the repository
git clone 

## download models and demo video
- download models from [google drive](https://drive.google.com/drive/folders/19PTZ7EBu1HPEjtqfXPJPjCyg3D7ANGkM?usp=sharing)
- download data from [google drive](https://drive.google.com/drive/folders/1rNqr1aYlHgu3kEXT7GTaTTpP4uP9xLoR?usp=sharing)
- copy the models and data folders in project directory

## build the container
docker build -t openvino.

## run docker container
docker run -it --network=host -v $PWD:/app openvino

## run fare evasion detection on a video
python Service.py --in_video_path data/videos/clips.avi --device CPU

A new experiment directory will be created in runs/ folder and result video and captured violations will be stored there.

## demo video tutorial






