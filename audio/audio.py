from pydub import AudioSegment
from pydub.playback import play
from utils import get_project_root


def play_training_is_complete():
    filename = str(get_project_root()) + "/audio/training_complete.mp3"
    song = AudioSegment.from_mp3(filename)
    play(song)
