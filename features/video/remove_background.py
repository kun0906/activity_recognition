
"""
pip3 install pixellib

download models: https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/xception_pascalvoc.pb

https://github.com/ayoolaolafenwa/PixelLib/releases/


"""
import pixellib
from pixellib.tune_bg import alter_bg

change_bg = alter_bg(model_type = "pb")
change_bg.load_pascalvoc_model("./features/xception_pascalvoc.pb")
# change_bg.color_video("sample_video.mp4", colors = (0,128,0), frames_per_second=10, output_video_name="output_video.mp4",
# detect = "person")

change_bg.color_video("data/ask_time_1_1614904536_1.mp4",
                      colors = (0,128,0),
                      frames_per_second=10,
                      output_video_name="out/output_video_1.mp4",
                      detect = "person")

