import os, urllib, sys


__all__ = [
    "download_file"
]


def download_file(url: str, filepath: str=None) -> str:
    def progress( block_count, block_size, total_size ):
        percentage = 100.0 * block_count * block_size / total_size
        # I don't want to use a new line, so I won't use a print statement.
        sys.stdout.write( "%.2f %% ( %d KB )\r" % ( percentage, total_size / 1024 ) )
    if filepath is None:
        filepath = "./" + os.path.basename(url)
    urllib.request.urlretrieve(url=url, filename=filepath, reporthook=progress)
    print("")
    return filepath
