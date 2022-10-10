import os
from pytube import YouTube

yt = YouTube('https://www.youtube.com/watch?v=shoVsQhou-8')

stream = yt.streams.get_by_itag(249)

data_path = os.path.join(os.getcwd(), 'music_data')
if os.path.isdir(data_path):
    os.mkdir(data_path)
stream.download(output_path='music_data')

print('done')