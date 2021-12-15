
## WORKING EXAMPLE USING FFMPEG
# import subprocess as sp
# import numpy as np
# import time
# # http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
# command = [ "ffmpeg",
#         '-y', # (optional) overwrite output file if it exists
#         '-f', 'rawvideo',
#         '-vcodec','rawvideo',
#         '-s', '420x360', # size of one frame
#         '-pix_fmt', 'rgb24',
#         '-r', '24', # frames per second
#         '-i', '-', # The imput comes from a pipe
#         '-an', # Tells FFMPEG not to expect any audio
#         '-vcodec', 'mpeg',
#         'my_output_videofile.mp4' ]
# command = ['ffmpeg',
#            '-loglevel', 'error',
#            '-y',
#            # Input
#            '-f', 'rawvideo',
#            '-vcodec', 'rawvideo',
#            '-pix_fmt', 'bgr24',
#         '-s', '420x360', # size of one frame
#         '-r', '24', # frames per second
#            # Output
#            '-i', '-',
#            '-an',
#            '-vcodec', 'h264',
# #            '-b:v', str(bitrate) + 'M',
#            '-pix_fmt', 'bgr24',
#         'my_output_videofile.mp4' ]

# pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)

# for _ in range(2):
#     pipe.stdin.write( np.zeros((420,360,3), np.uint8).tobytes() )
#     time.sleep(0.1)
# pipe.stdin.close()
# pipe.wait()
