# python "F:\theotsmp4.py"

# tsurl=[https://citcastmedia.nus.edu.sg/vod/_definst_/mp4:pivle2/users/a6/51/a6512787-5251-4a72-bf65-f4704bd7f4be.mp4/playlist.m3u8]
tsurl=['https://citcastmedia.nus.edu.sg/vod/_definst_/mp4:pivle2/users/a6/51/a6512787-5251-4a72-bf65-f4704bd7f4be.mp4/media_w1562646809_3.ts']
outname=['1102-0-1']
# 'E:/totube/1102-0-1.mp4'

# folder ='F:\\nus\y1\MA1102R\0-3'
import subprocess

for x in range(len(tsurl)):
    plst=tsurl[x].rindex('/')
    url=tsurl[x][:plst]+'/playlist.m3u8'
    out='E:/totube/'+outname[x]+'.mp4'
    fin='E:/totube/'+outname[x]+'cary.mp4'
    # print(url,out)
    subprocess.run(['ffmpeg.exe', '-i',url , '-c', 'copy', out])
    subprocess.run(['python', 'f:jumpcutedit.py', '--input_file', out, '--output_file', fin, '--sounded_speed', '1.5', '--silent_speed', '5', '--frame_margin', '2', '--frame_rate', '30'])



# https://citcastmedia.nus.edu.sg/vod/_definst_/mp4:pivle2/users/84/0c/840c9892-29b4-489e-942f-e353e68c0467.mp4/_3.ts
# ffmpeg.exe -i https://citcastmedia.nus.edu.sg/vod/_definst_/mp4:pivle2/users/a6/51/a6512787-5251-4a72-bf65-f4704bd7f4be.mp4/playlist.m3u8 -c copy E:/totube/output.mp4
# subprocess.run(['ffmpeg.exe', '-i', 'https://citcastmedia.nus.edu.sg/vod/_definst_/mp4:pivle2/users/a6/51/a6512787-5251-4a72-bf65-f4704bd7f4be.mp4/playlist.m3u8', '-c', 'copy', 'E:/totube/outputi.mp4'])
# subprocess.run(['ffmpeg.exe', '-i',tsurl[x] , '-c', 'copy', ])
# subprocess.run(['ffmpeg', '-i', infile, outfile])
# subprocess.run([python f:jumpcutter.py --input_file E:\totube\1102-0-1.mp4 --sounded_speed 1.5 --silent_speed 5 --frame_margin 2 --frameRate 30])
# subprocess.run(['python', 'f:jumpcutedit.py', '--input_file', 'E:\\totube\\1102-0-1.mp4', '--output_file', 'E:\\totube\\1102-0-1cary.mp4', '--sounded_speed', '1.5', '--silent_speed', '5', '--frame_margin', '2', '--frame_rate', '30'])

# TEMP_FOLDER = "TEMP"
# directory = TEMP_FOLDER
# parent_dir = "E:/totube"#"/home/User/Documents"
# path = os.path.join(parent_dir, directory)
# os.mkdir(path)

# from os import listdir
# from os.path import isfile, join
#
# # allfiles = [f for f in listdir(folder) if isfile(join(folder, f))] # list of '01-PedalOffForte1Close.flac'
# # note=[int(x[:1]) for x in allfiles] # 01
#
# tsfiles = [f for f in listdir(folder)]
# print('ts',tsfiles)
# # https://superuser.com/questions/692990/use-ffmpeg-copy-codec-to-combine-ts-files-into-a-single-mp4
# # copy /b segment1_0_av.ts+segment2_0_av.ts+segment3_0_av.ts all.ts
# # ffmpeg -i all.ts -acodec copy -vcodec copy all.mp4
#
# # from distutils.dir_util import copy_tree
# # copy_tree("/a/b/c", "/x/y/z")
# # copy /b _0.ts+_1.ts+_2.ts+_3.ts all.ts
# # copy /b '_0.ts'+'_1.ts'+'_2.ts'+'_3.ts' all.ts
# # copy /b 'F:\nus\y1\MA1102R\0-3\_0.ts'+'F:\nus\y1\MA1102R\0-3\_1.ts' all.ts
# # copy /b \\nus\y1\MA1102R\0-3\_0.ts+\\nus\y1\MA1102R\0-3\_1.ts all.ts
# # copy /b tsfiles all.ts
# # ffmpeg -i all.ts -acodec copy -vcodec copy all.mp4
#
# # https://www.codementor.io/@chuksdurugo/download-and-combine-media-segments-of-a-hls-stream-locally-using-ffmpeg-150zo6t775
# # ffmpeg -i https://mnmedias.api.telequebec.tv/m3u8/29880.m3u8 -map p:6 -c copy -t 60 -f segment -segment_list out.list out%03d.ts
# ffmpeg -f concat -safe 0 -i <(for f in ./*.ts; do echo "file '$PWD/$f'"; done) -c copy playlist.mp4
# # https://stackoverflow.com/questions/37105973/converting-ts-to-mp4-with-python
