"""
From odat.mdat.phone_digits written for this app
"""

from py2store import wrap_kvs, filt_iter, FilesOfZip
import soundfile as sf
from io import BytesIO
from slang import KvDataSource
import re
import numpy as np
from sklearn.preprocessing import normalize


@wrap_kvs(obj_of_data=lambda b: sf.read(BytesIO(b), dtype="int16")[0])
@filt_iter(
    filt=lambda x: not x.startswith("__MACOSX")
    and x.startswith("train/")
    and x.endswith(".wav")
)
class WfStore(FilesOfZip):
    pass


def mk_dacc(zip_dir):
    return Dacc(zip_dir=zip_dir)


def mk_ds(zip_dir):
    s = WfStore(zip_dir)

    path_component = re.compile("[^_]+")

    def key(x):
        m = path_component.match(x)
        if m:
            return int(m.group(0)[-1])

    ds = KvDataSource(s, key_to_tag=key)
    return ds


class Dacc:
    def __init__(self, zip_dir):
        self.ds = mk_ds(zip_dir)

    def wf_tag_gen(self):
        for _, tag, wf in self.ds.key_tag_wf_gen():
            normal_wf = normalize(np.float32(wf).reshape(1, -1))[0]
            yield normal_wf, tag

    def chk_tag_gen(self, chunker):
        for wf, tag in self.wf_tag_gen():
            for chk in chunker(wf):
                yield chk, tag

    def snips_tag_gen(self, snipper):
        for wf, tag in self.wf_tag_gen():
            yield snipper.wf_to_snips(wf), tag
