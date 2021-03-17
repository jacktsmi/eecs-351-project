from savify import Savify
from savify.types import Type, Format, Quality


def load_playlist(mood='happy'):
    s = Savify()
    #s = Savify(api_credentials=(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET))
    s.download(mood, query_type=Type.PLAYLIST)

#s = Savify(api_credentials=(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET))
load_playlist('happy')